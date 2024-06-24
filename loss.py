import tensorflow as tf

@tf.function
def modified_mae(y_true, y_pred):
    """
    Calculates the mean absolute error between the true and predicted values.
    
    Parameters:
    y_true (Tensor): The true values.
    y_pred (Tensor): The predicted values.
    
    Returns:
    float: The mean absolute error.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    s = tf.constant(0.0, dtype=tf.float32)
    cnt = tf.constant(0.0, dtype=tf.float32)
    
    for i in range(len(y_true)):
        if (tf.reduce_sum(y_true[i]) != 0):
            s += tf.reduce_sum(abs(y_true[i] - y_pred[i]))
            cnt += 1.0
    if cnt > 0.0:
        s /= cnt
    return s

@tf.function
def modified_mse(y_true, y_pred):
    """
    Calculates the mean squared error (MSE) between the true labels (y_true) and the predicted labels (y_pred).

    Parameters:
    - y_true: The true labels.
    - y_pred: The predicted labels.

    Returns:
    - The mean squared error between y_true and y_pred.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    s = tf.constant(0.0, dtype=tf.float32)
    cnt = tf.constant(0.0, dtype=tf.float32)
    
    for i in tf.range(len(y_true)):
        if tf.reduce_sum(y_true[i]) != 0:
            s += tf.reduce_sum(tf.square(y_true[i] - y_pred[i]))
            cnt += 1.0
    
    if cnt > 0.0:
        s /= cnt
    return s

@tf.function
def modified_categorical_crossentropy(y_true, y_pred):
    m = y_true.shape[0]
    loss = tf.zeros([])
    for i in range(m):
        for j in range(y_true[i].shape[0]):
            loss += tf.keras.losses.CategoricalCrossentropy()(y_true[i][j], y_pred[i][j])

    loss /= tf.cast(m, tf.float32)
    return loss

c = tf.keras.metrics.CategoricalAccuracy()

@tf.function
def modified_accuracy(y_true, y_pred):
    m = y_true.shape[0]
    total_accuracy = tf.constant(0.0, dtype=tf.float32)
    
    for i in range(m):
        cnt = tf.constant(0, dtype=tf.float32)
        accuracy_sum = tf.constant(0.0, dtype=tf.float32)
        
        for j in range(tf.shape(y_true[i])[0]):
            mask = tf.logical_or(
                tf.not_equal(tf.argmax(y_true[i][j], axis=-1), 11),  # Exclude class 12 (index 11)
                tf.not_equal(tf.argmax(y_true[i][j], axis=-1), 12)   # Exclude class 13 (index 12)
            )
            
            if tf.reduce_all(mask):  # Ensure that all elements are valid
                c.update_state(y_true[i][j], y_pred[i][j])
                accuracy_sum += c.result()
                cnt += 1

        if cnt > 0:  # Ensure we don't divide by zero
            total_accuracy += accuracy_sum / cnt
    
    total_accuracy /= tf.cast(m, tf.float32)
    return total_accuracy
