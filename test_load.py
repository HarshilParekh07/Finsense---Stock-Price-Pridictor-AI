import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
import traceback

class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        print(f"DEBUG: PatchedInputLayer called with kwargs keys: {list(kwargs.keys())}")
        kwargs.pop('optional', None)
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        
        # Aggressively clean up dtype if it's a dict (Keras 3 policy)
        if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
            print(f"DEBUG: Cleaning up complex dtype: {kwargs['dtype']}")
            kwargs['dtype'] = 'float32' # Default to float32
            
        try:
            super().__init__(*args, **kwargs)
        except TypeError as e:
            print(f"DEBUG: super().__init__ failed with TypeError: {e}")
            # Try even more minimal init
            minimal_kwargs = {
                'batch_input_shape': kwargs.get('batch_input_shape'),
                'name': kwargs.get('name')
            }
            super().__init__(**minimal_kwargs)

print("\n--- Trying patched load of keras_model.h5 ---")
try:
    model = load_model('keras_model.h5', custom_objects={'InputLayer': PatchedInputLayer}, compile=False)
    print("Patched load success")
except Exception:
    traceback.print_exc()
