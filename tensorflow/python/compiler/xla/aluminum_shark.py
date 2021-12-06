# this is part of tensorflow for now but in the future we should move that so
# we can build this indepently of tensorflow.
# this is the python entrypoint to the world of aluminum_shark

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid
import warnings
import os

from tensorflow.python import _pywrap_aluminum_shark as wrap
from tensorflow.python.util.tf_export import tf_export


# Wrap the impl classes very thinly so we can export it
@tf_export("aluminum_shark.CipherText")
class CipherText(object):

  def __init__(self, handle: wrap.CipherTextHandleImpl) -> None:
    self.__handle = handle

  @property
  def handle(self):
    return self.__handle


@tf_export("aluminum_shark.Context")
class Context(object):

  def __init__(self, handle: wrap.ContextImpl) -> None:
    self.__handle = handle

  def encrypt(self, ptxt, handle=None):
    if handle is not None:
      warnings.warn("user specified handles should only be used for debugging")
    else:
      handle = str(uuid.uuid1())
    return self.__handle.encrypt(ptxt, handle)

  def decrypt(self, ctxt):
    return self.__handle.decrypt(ctxt.handle)


@tf_export("aluminum_shark.create_context")
def create_context():
  return Context(wrap.ContextImpl())


@tf_export("aluminum_shark.debug_on")
def debug_on(flag):
  os.environ['ALUMINUM_SHARK_LOGGING'] = "1" if flag else "0"


@tf_export("aluminum_shark.set_ciphertexts")
def set_ciphertexts(ctxt: CipherText):
  print("aluminum_shark.set_ciphertexts")
  wrap.setCiphertext(ctxt)


@tf_export("aluminum_shark.get_ciphertexts")
def get_ciphertexts(ctxt):
  print("aluminum_shark.get_ciphertexts")
  return CipherText(wrap.getCiphertext(ctxt))
