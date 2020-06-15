from mlflow import log_metrics
from tensorflow.keras.callbacks import Callback

class MlFlowCallback(Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        log_metrics(logs, step=epoch)
