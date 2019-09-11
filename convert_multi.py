import tensorflow as tf
import keras
from core_multi.models import MultiStyleTransferNetwork


class SavedModelConvertMulti(object):

    @classmethod
    def convert_multi_to_saved_model(cls):
        # start to convert
        export_path = 'savedModels/model_1'
        model_path = '***.h5'

        keras.backend.clear_session()
        keras.backend.set_learning_phase(0)

        # params need to be changed according to your own model
        model = MultiStyleTransferNetwork.build(
            (256, 256),
            47,
            alpha=0.5,
            checkpoint_file=model_path
        )

        # the key can be defined by your self
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={'image': model.input}, outputs={'output_image': model.output})

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess=keras.backend.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature
            }
        )

        builder.save()


SavedModelConvertMulti.convert_multi_to_saved_model()
