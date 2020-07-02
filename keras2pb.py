import tensorflow as tf
import os
import VGG
import keys_560W
from keras.models import Model
from keras.layers import Input
from tensorflow.python.framework import graph_io

characters = keys_560W.CHAR_ALL_560W[0:]
nclass = len(characters)

def keras_model_to_frozen_graph():
    """ convert keras h5 model file to frozen graph(.pb file)
    """
    def freeze_graph(graph, session, output_node_names, model_name):
        with graph.as_default():
            graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
            graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output_node_names)
            graph_io.write_graph(graphdef_frozen, "models", os.path.basename(model_name) + ".pb", as_text=False)

    tf.keras.backend.set_learning_phase(0) # this line most important

    input = Input(shape=(32, None, 1), name='the_input')
    y_pred = VGG.VGG_cnn(input, nclass, use_LSTM=False)
    #basemodel = VGG.VGG_cnn(input, nclass, use_LSTM=False)
    model = Model(inputs=input, outputs=y_pred)
    #model = load_net()
    model_path = './models/weights_vggnet_560w-03-32-0.47_138.h5'
    model_name = model_path.split('/')[-1].split('.h5')[0]
    model.load_weights(model_path)
    session = tf.keras.backend.get_session()
    freeze_graph(session.graph, session, [out.op.name for out in model.outputs], model_name)

if __name__ == '__main__':
    keras_model_to_frozen_graph()
