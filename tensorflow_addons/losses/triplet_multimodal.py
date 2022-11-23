# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements multimodal triplet loss."""

import tensorflow as tf
from .metric_learning import *
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from typeguard import typechecked
from typing import Optional, Union, Callable


def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of shape `[n, m]`.
      mask: 2-D Boolean `Tensor` of shape `[n, m]`.
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums


def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of shape `[n, m]`.
      mask: 2-D Boolean `Tensor` of shape `[n, m]`.
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def multimodal_triplet_semihard_loss_img_to_text(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    r"""Computes the multimodal triplet loss with semi-hard negative mining.
    Usage:
    >>> y_true = tf.convert_to_tensor([0, 0])
    >>> y_pred = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    >>> tfa.losses.triplet_semihard_loss(y_true, y_pred, distance_metric="L2")
    <tf.Tensor: shape=(), dtype=float32, numpy=2.4142137>
    >>> # Calling with callable `distance_metric`
    >>> distance_metric = lambda x: tf.linalg.matmul(x, x, transpose_b=True)
    >>> tfa.losses.triplet_semihard_loss(y_true, y_pred, distance_metric=distance_metric)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of concatenated image and text embedding vectors with shape `[batch_size, img_embeddings+text_embeddings]`. Embeddings should
        be l2 normalized. Image and text embeddings should have the same shape.
      margin: Float, margin term in the loss definition.
      distance_metric: `str` or a `Callable` that determines distance metric.
        Valid strings are "L2" for l2-norm distance,
        "squared-L2" for squared l2-norm distance,
        and "angular" for cosine similarity.
        A `Callable` should take a batch of embeddings as input and
        return the pairwise distance matrix.
    Returns:
      triplet_loss: float scalar with dtype of `y_pred`.
    """
    
    labels = y_true
    embeddings_img = y_pred[:, :(y_pred.shape[1] // 2)]
    embeddings_text = y_pred[:, (y_pred.shape[1] // 2):y_pred.shape[1]]

    convert_to_float32_img = (
        embeddings_img.dtype == tf.dtypes.float16 or embeddings_img.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings_img = (
        tf.cast(embeddings_img, tf.dtypes.float32) if convert_to_float32_img else embeddings_img
    )
    convert_to_float32_text = (
        embeddings_text.dtype == tf.dtypes.float16 or embeddings_text.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings_text = (
        tf.cast(embeddings_text, tf.dtypes.float32) if convert_to_float32_text else embeddings_text
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])
    
    # Build pairwise squared distance matrix

    if distance_metric == "L2":
        pdist_matrix = multimodal_pairwise_distance(
            precise_embeddings_img, precise_embeddings_text, squared=False
        )

    elif distance_metric == "squared-L2":
        pdist_matrix = multimodal_pairwise_distance(
            precise_embeddings_img, precise_embeddings_text, squared=True
        )

    elif distance_metric == "angular":
        pdist_matrix = multimodal_angular_distance(
            precise_embeddings_img, precise_embeddings_text
        )
    
    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)

    # Compute the mask.
    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
    mask = tf.math.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.math.greater(
            pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])
        ),
    )
    mask_final = tf.reshape(
        tf.math.greater(
            tf.math.reduce_sum(
                tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True
            ),
            0.0,
        ),
        [batch_size, batch_size],
    )
    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    mask = tf.cast(mask, dtype=tf.dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = tf.reshape(
        _masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
    )
    negatives_outside = tf.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = tf.tile(
        _masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
    )
    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = tf.math.reduce_sum(mask_positives)

    triplet_loss = tf.math.truediv(
        tf.math.reduce_sum(
            tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
        ),
        num_positives,
    )

    if convert_to_float32_img:
        return tf.cast(triplet_loss, embeddings_img.dtype)
    elif convert_to_float32_text:
        return tf.cast(triplet_loss, embeddings_text.dtype)
    else:
        return triplet_loss


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def multimodal_triplet_semihard_loss_text_to_img(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    r"""Computes the multimodal triplet loss with semi-hard negative mining.
    Usage:
    >>> y_true = tf.convert_to_tensor([0, 0])
    >>> y_pred = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    >>> tfa.losses.triplet_semihard_loss(y_true, y_pred, distance_metric="L2")
    <tf.Tensor: shape=(), dtype=float32, numpy=2.4142137>
    >>> # Calling with callable `distance_metric`
    >>> distance_metric = lambda x: tf.linalg.matmul(x, x, transpose_b=True)
    >>> tfa.losses.triplet_semihard_loss(y_true, y_pred, distance_metric=distance_metric)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of concatenated image and text embedding vectors with shape `[batch_size, img_embeddings+text_embeddings]`. Embeddings should
        be l2 normalized. Image and text embeddings should have the same shape.
      margin: Float, margin term in the loss definition.
      distance_metric: `str` or a `Callable` that determines distance metric.
        Valid strings are "L2" for l2-norm distance,
        "squared-L2" for squared l2-norm distance,
        and "angular" for cosine similarity.
        A `Callable` should take a batch of embeddings as input and
        return the pairwise distance matrix.
    Returns:
      triplet_loss: float scalar with dtype of `y_pred`.
    """
    
    labels = y_true
    embeddings_img = y_pred[:, :(y_pred.shape[1] // 2)]
    embeddings_text = y_pred[:, (y_pred.shape[1] // 2):y_pred.shape[1]]

    convert_to_float32_img = (
        embeddings_img.dtype == tf.dtypes.float16 or embeddings_img.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings_img = (
        tf.cast(embeddings_img, tf.dtypes.float32) if convert_to_float32_img else embeddings_img
    )
    convert_to_float32_text = (
        embeddings_text.dtype == tf.dtypes.float16 or embeddings_text.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings_text = (
        tf.cast(embeddings_text, tf.dtypes.float32) if convert_to_float32_text else embeddings_text
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])
    
    # Build pairwise squared distance matrix

    if distance_metric == "L2":
        pdist_matrix = multimodal_pairwise_distance(
            precise_embeddings_text, precise_embeddings_img, squared=False
        )

    elif distance_metric == "squared-L2":
        pdist_matrix = multimodal_pairwise_distance(
            precise_embeddings_text, precise_embeddings_img, squared=True
        )

    elif distance_metric == "angular":
        pdist_matrix = multimodal_angular_distance(
            precise_embeddings_text, precise_embeddings_img
        )
    
    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)

    # Compute the mask.
    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
    mask = tf.math.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.math.greater(
            pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])
        ),
    )
    mask_final = tf.reshape(
        tf.math.greater(
            tf.math.reduce_sum(
                tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True
            ),
            0.0,
        ),
        [batch_size, batch_size],
    )
    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    mask = tf.cast(mask, dtype=tf.dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = tf.reshape(
        _masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
    )
    negatives_outside = tf.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = tf.tile(
        _masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
    )
    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = tf.math.reduce_sum(mask_positives)

    triplet_loss = tf.math.truediv(
        tf.math.reduce_sum(
            tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
        ),
        num_positives,
    )

    if convert_to_float32_img:
        return tf.cast(triplet_loss, embeddings_img.dtype)
    elif convert_to_float32_text:
        return tf.cast(triplet_loss, embeddings_text.dtype)
    else:
        return triplet_loss


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def multimodal_triplet_semihard_loss_bidirectional(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    lambda_1: FloatTensorLike = 0.05,
    lambda_2: FloatTensorLike = 0.05,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    return lambda_1 * multimodal_triplet_semihard_loss_text_to_img(y_true, y_pred, margin, distance_metric) + lambda_2 * multimodal_triplet_semihard_loss_img_to_text(y_true, y_pred, margin, distance_metric)


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def multimodal_triplet_hard_loss_img_to_text(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    soft: bool = False,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    r"""Computes the triplet loss with hard negative and hard positive mining.
    Usage:
    >>> y_true = tf.convert_to_tensor([0, 0])
    >>> y_pred = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    >>> tfa.losses.triplet_hard_loss(y_true, y_pred, distance_metric="L2")
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    >>> # Calling with callable `distance_metric`
    >>> distance_metric = lambda x: tf.linalg.matmul(x, x, transpose_b=True)
    >>> tfa.losses.triplet_hard_loss(y_true, y_pred, distance_metric=distance_metric)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
      soft: Boolean, if set, use the soft margin version.
      distance_metric: `str` or a `Callable` that determines distance metric.
        Valid strings are "L2" for l2-norm distance,
        "squared-L2" for squared l2-norm distance,
        and "angular" for cosine similarity.
        A `Callable` should take a batch of embeddings as input and
        return the pairwise distance matrix.
    Returns:
      triplet_loss: float scalar with dtype of `y_pred`.
    """

    labels = y_true
    embeddings_img = y_pred[:, :(y_pred.shape[1] // 2)]
    embeddings_text = y_pred[:, (y_pred.shape[1] // 2):y_pred.shape[1]]

    convert_to_float32_img = (
        embeddings_img.dtype == tf.dtypes.float16 or embeddings_img.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings_img = (
        tf.cast(embeddings_img, tf.dtypes.float32) if convert_to_float32_img else embeddings_img
    )
    convert_to_float32_text = (
        embeddings_text.dtype == tf.dtypes.float16 or embeddings_text.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings_text = (
        tf.cast(embeddings_text, tf.dtypes.float32) if convert_to_float32_text else embeddings_text
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])
    
    # Build pairwise squared distance matrix

    if distance_metric == "L2":
        pdist_matrix = multimodal_pairwise_distance(
            precise_embeddings_img, precise_embeddings_text, squared=False
        )

    elif distance_metric == "squared-L2":
        pdist_matrix = multimodal_pairwise_distance(
            precise_embeddings_img, precise_embeddings_text, squared=True
        )

    elif distance_metric == "angular":
        pdist_matrix = multimodal_angular_distance(
            precise_embeddings_img, precise_embeddings_text
        )

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    # hard negatives: smallest D_an.
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = tf.size(labels)

    adjacency = tf.cast(adjacency, dtype=tf.dtypes.float32)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # hard positives: largest D_ap.
    hard_positives = _masked_maximum(pdist_matrix, mask_positives)

    if soft:
        triplet_loss = tf.math.log1p(tf.math.exp(hard_positives - hard_negatives))
    else:
        triplet_loss = tf.maximum(hard_positives - hard_negatives + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def multimodal_triplet_hard_loss_text_to_img(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    soft: bool = False,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    r"""Computes the triplet loss with hard negative and hard positive mining.
    Usage:
    >>> y_true = tf.convert_to_tensor([0, 0])
    >>> y_pred = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    >>> tfa.losses.triplet_hard_loss(y_true, y_pred, distance_metric="L2")
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    >>> # Calling with callable `distance_metric`
    >>> distance_metric = lambda x: tf.linalg.matmul(x, x, transpose_b=True)
    >>> tfa.losses.triplet_hard_loss(y_true, y_pred, distance_metric=distance_metric)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
      soft: Boolean, if set, use the soft margin version.
      distance_metric: `str` or a `Callable` that determines distance metric.
        Valid strings are "L2" for l2-norm distance,
        "squared-L2" for squared l2-norm distance,
        and "angular" for cosine similarity.
        A `Callable` should take a batch of embeddings as input and
        return the pairwise distance matrix.
    Returns:
      triplet_loss: float scalar with dtype of `y_pred`.
    """

    labels = y_true
    embeddings_img = y_pred[:, :(y_pred.shape[1] // 2)]
    embeddings_text = y_pred[:, (y_pred.shape[1] // 2):y_pred.shape[1]]

    convert_to_float32_img = (
        embeddings_img.dtype == tf.dtypes.float16 or embeddings_img.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings_img = (
        tf.cast(embeddings_img, tf.dtypes.float32) if convert_to_float32_img else embeddings_img
    )
    convert_to_float32_text = (
        embeddings_text.dtype == tf.dtypes.float16 or embeddings_text.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings_text = (
        tf.cast(embeddings_text, tf.dtypes.float32) if convert_to_float32_text else embeddings_text
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])
    
    # Build pairwise squared distance matrix

    if distance_metric == "L2":
        pdist_matrix = multimodal_pairwise_distance(
            precise_embeddings_text, precise_embeddings_img, squared=False
        )

    elif distance_metric == "squared-L2":
        pdist_matrix = multimodal_pairwise_distance(
            precise_embeddings_text, precise_embeddings_img, squared=True
        )

    elif distance_metric == "angular":
        pdist_matrix = multimodal_angular_distance(
            precise_embeddings_text, precise_embeddings_img
        )

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    # hard negatives: smallest D_an.
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = tf.size(labels)

    adjacency = tf.cast(adjacency, dtype=tf.dtypes.float32)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # hard positives: largest D_ap.
    hard_positives = _masked_maximum(pdist_matrix, mask_positives)

    if soft:
        triplet_loss = tf.math.log1p(tf.math.exp(hard_positives - hard_negatives))
    else:
        triplet_loss = tf.maximum(hard_positives - hard_negatives + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def multimodal_triplet_hard_loss_bidirectional(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    lambda_1: FloatTensorLike = 0.05,
    lambda_2: FloatTensorLike = 0.05,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    return lambda_1 * multimodal_triplet_hard_loss_text_to_img(y_true, y_pred, margin, distance_metric) + lambda_2 * multimodal_triplet_hard_loss_img_to_text(y_true, y_pred, margin, distance_metric)


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultimodalTripletSemihardLossImgToText(LossFunctionWrapper):
    """Computes the unidirectional triplet loss with semi-hard negative mining.

    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance
    among which are at least greater than the positive distance plus the
    margin constant (called semi-hard negative) in the mini-batch.
    If no such negative exists, uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.

    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    `[batch_size]` of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.

    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            multimodal_triplet_semihard_loss_img_to_text,
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            margin=margin,
            distance_metric=distance_metric,
        )


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultimodalTripletSemihardLossTextToImg(LossFunctionWrapper):
    """Computes the unidirectional triplet loss with semi-hard negative mining.

    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance
    among which are at least greater than the positive distance plus the
    margin constant (called semi-hard negative) in the mini-batch.
    If no such negative exists, uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.

    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    `[batch_size]` of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.

    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            multimodal_triplet_semihard_loss_text_to_img,
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            margin=margin,
            distance_metric=distance_metric,
        )


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultimodalTripletSemihardLossBidirectional(LossFunctionWrapper):
    """Computes the bidirectional triplet loss with semi-hard negative mining.

    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance
    among which are at least greater than the positive distance plus the
    margin constant (called semi-hard negative) in the mini-batch.
    If no such negative exists, uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.

    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    `[batch_size]` of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.

    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        lambda_1: FloatTensorLike = 0.05,
        lambda_2: FloatTensorLike = 0.05,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            multimodal_triplet_semihard_loss_bidirectional,
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            margin=margin,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            distance_metric=distance_metric,
        )


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultimodalTripletHardLossImgToText(LossFunctionWrapper):
    """Computes the unidirectional triplet loss with hard negative and hard positive mining.

    The loss encourages the maximum positive distance (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance plus the
    margin constant in the mini-batch.
    The loss selects the hardest positive and the hardest negative samples
    within the batch when forming the triplets for computing the loss.
    See: https://arxiv.org/pdf/1703.07737.

    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    `[batch_size]` of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.

    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      soft: Boolean, if set, use the soft margin version. Default value is False.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        soft: bool = False,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            multimodal_triplet_hard_loss_img_to_text,
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            margin=margin,
            soft=soft,
            distance_metric=distance_metric,
        )


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultimodalTripletHardLossTextToImg(LossFunctionWrapper):
    """Computes the unidirectional triplet loss with hard negative and hard positive mining.

    The loss encourages the maximum positive distance (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance plus the
    margin constant in the mini-batch.
    The loss selects the hardest positive and the hardest negative samples
    within the batch when forming the triplets for computing the loss.
    See: https://arxiv.org/pdf/1703.07737.

    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    `[batch_size]` of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.

    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      soft: Boolean, if set, use the soft margin version. Default value is False.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        soft: bool = False,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            multimodal_triplet_hard_loss_text_to_img,
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            margin=margin,
            soft=soft,
            distance_metric=distance_metric,
        )


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultimodalTripletHardLossBidirectional(LossFunctionWrapper):
    """Computes the bidirectional triplet loss with hard negative and hard positive mining.

    The loss encourages the maximum positive distance (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance plus the
    margin constant in the mini-batch.
    The loss selects the hardest positive and the hardest negative samples
    within the batch when forming the triplets for computing the loss.
    See: https://arxiv.org/pdf/1703.07737.

    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    `[batch_size]` of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.

    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      soft: Boolean, if set, use the soft margin version. Default value is False.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        lambda_1: FloatTensorLike = 0.05,
        lambda_2: FloatTensorLike = 0.05,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            multimodal_triplet_hard_loss_bidirectional,
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            margin=margin,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            distance_metric=distance_metric,
        )