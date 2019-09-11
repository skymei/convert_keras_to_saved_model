import os
import keras
import tensorflow as tf
from keras import backend as K
from core_single.models import StyleTransferNetwork


class SavedModelConvertSingle(object):

    @classmethod
    def convert_single_to_saved_model(cls):
        models = [os.path.basename(x) for x in os.listdir('single_models')]

        for k, model in enumerate(models):
            model_path = os.path.join('single_models', model)
            output_dir = f"output/single_test_model_{k+1}/1"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, 0o755)

            # start to convert
            keras.backend.clear_session()
            keras.backend.set_learning_phase(0)

            # alpha is defined by model
            model = StyleTransferNetwork.build((None, None), alpha=0.5)
            model.load_weights(model_path, by_name=False)

            tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
            signature = tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={'image': model.input}, outputs={'output_image': model.output})

            builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
            builder.add_meta_graph_and_variables(
                sess=K.get_session(),
                tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        signature
                })

            builder.save()


SavedModelConvertSingle().convert_single_to_saved_model()
