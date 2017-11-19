
import tensorflow as tf
from tflearn.collections import *

class _DumpableGraph(object):
    """The dumpable graph
    """
    Keys = [ "summary_tags", tf.GraphKeys.GRAPH_CONFIG, tf.GraphKeys.DATA_PREP, tf.GraphKeys.DATA_AUG, tf.GraphKeys.TRAIN_OPS ] #pylint: disable=I0011,E1101

    def __init__(self):
        """Create a new _DumpableGraph
        """
        self._values = {}

    def __enter__(self):
        """Make this graph dumpable
        """
        for key in self.Keys:
            values = tf.get_collection_ref(key)
            self._values[key] = list(values)
            del values[:]

    def __exit__(self, _, __, ___):
        """Restore this graph
        """
        for key in self.Keys:
            if key not in self._values:
                continue
            tf.get_collection_ref(key).extend(self._values[key])

tf.logging.set_verbosity(tf.logging.INFO)

def parse_csv(rows_string_tensor, csv_columns, csv_column_defaults, unused_columns=None):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""

  unused_columns = unused_columns or []
  # Takes a rank-1 tensor and converts it into rank-2 tensor
  # Example if the data is ['csv,line,1', 'csv,line,2', ..] to
  # [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
  # tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
  row_columns = tf.expand_dims(rows_string_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=csv_column_defaults)
  features = dict(zip(csv_columns, columns))

  # Remove unused columns
  for col in unused_columns:
    features.pop(col)
  return features

def csv_serving_input_fn(csv_columns, csv_column_defaults, label_col=None,
                         default_batch_size=None, unused_columns=None):
  """Build the serving inputs.

  Args:
    default_batch_size (int): Batch size for the tf.placeholder shape
  """
  csv_row = tf.placeholder(
      shape=[default_batch_size],
      dtype=tf.string
  )
  features = parse_csv(csv_row, csv_columns, csv_column_defaults, unused_columns)
  if label_col:
    features.pop(label_col)
  return features, {'csv_row': csv_row}

def json_serving_input_fn(continuous_cols=None, categorical_cols=None, default_batch_size=None):
  """Build the serving inputs.

  Args:
    default_batch_size (int): Batch size for the tf.placeholder shape
  """
  inputs = {}
  continuous_cols = continuous_cols or {}
  categorical_cols = categorical_cols or {}

  for feat, place_holder in continuous_cols.iteritems():
    # print("\nFEAT: %s  place_holder: %s\n" % (feat, place_holder))
    inputs[feat] = place_holder

  for feat, place_holder in categorical_cols.iteritems():
    inputs[feat] = place_holder

  features = {
      key: tf.expand_dims(tensor, -1) for key, tensor in inputs.iteritems()
  }
  return features, inputs

def build_and_run_exports(session, export_path, serving_input_fn, prediction_tensor, confidence_tensor=None,
                          latest=None, label_col = None, continuous_cols=None, categorical_cols=None,
                          csv_columns=None, csv_column_defaults=None, unused_columns=None):
  """Given the latest checkpoint file export the saved model.

  Args:
    latest (string): Latest checkpoint file
    export_path (string): Location of checkpoints and model files
    name (string): Name of the checkpoint to be exported. Used in building the
      export path.
    hidden_units (list): Number of hidden units
    learning_rate (float): Learning rate for the SGD
  """

  with session.graph.as_default(), _DumpableGraph():
      exporter = tf.saved_model.builder.SavedModelBuilder(export_dir=export_path)

      # print "(build_and_run_exports) continuous cols: ", continuous_cols
      features, inputs_dict = serving_input_fn(continuous_cols=continuous_cols,
                                                   categorical_cols=categorical_cols)

      prediction_dict = {'predictions': prediction_tensor}
      if confidence_tensor:
          prediction_dict['confidence'] = confidence_tensor
      # saver = tf.train.Saver()
      inputs_info = {
          name: tf.saved_model.utils.build_tensor_info(tensor) for name, tensor in inputs_dict.iteritems()
          }
      output_info = {
          name: tf.saved_model.utils.build_tensor_info(tensor) for name, tensor in prediction_dict.iteritems()
          }

      signature_def = tf.saved_model.signature_def_utils.build_signature_def(
          inputs=inputs_info,
          outputs=output_info,
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      )
      # print("\nsignature_def: %s" % signature_def)

      exporter.add_meta_graph_and_variables(
          session,
          tags=[tf.saved_model.tag_constants.SERVING],
          signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def},
      )

      exporter.save()
