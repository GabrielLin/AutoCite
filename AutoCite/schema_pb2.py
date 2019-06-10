import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='citeomatic/schema.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\x17\x63iteomatic/schema.proto\"\xd5\x01\n\x08\x44ocument\x12\r\n\x05title\x18\x01 \x01(\t\x12\x10\n\x08\x61\x62stract\x18\x02 \x01(\t\x12\x0f\n\x07\x61uthors\x18\x03 \x03(\t\x12\x15\n\rout_citations\x18\x04 \x03(\t\x12\x19\n\x11in_citation_count\x18\x05 \x01(\x05\x12\x0c\n\x04year\x18\x06 \x01(\x05\x12\n\n\x02id\x18\x07 \x01(\t\x12\r\n\x05venue\x18\x08 \x01(\t\x12\x13\n\x0bkey_phrases\x18\t \x03(\t\x12\x11\n\ttitle_raw\x18\n \x01(\t\x12\x14\n\x0c\x61\x62stract_raw\x18\x0b \x01(\t\" \n\x03Hit\x12\r\n\x05\x64ocid\x18\x01 \x01(\t\x12\n\n\x02tf\x18\x02 \x01(\x05\"!\n\x0bPostingList\x12\x12\n\x04hits\x18\x01 \x03(\x0b\x32\x04.Hitb\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_DOCUMENT = _descriptor.Descriptor(
  name='Document',
  full_name='Document',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='title', full_name='Document.title', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='abstract', full_name='Document.abstract', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='authors', full_name='Document.authors', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='out_citations', full_name='Document.out_citations', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='in_citation_count', full_name='Document.in_citation_count', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='year', full_name='Document.year', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='id', full_name='Document.id', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='venue', full_name='Document.venue', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='key_phrases', full_name='Document.key_phrases', index=8,
      number=9, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='title_raw', full_name='Document.title_raw', index=9,
      number=10, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='abstract_raw', full_name='Document.abstract_raw', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28,
  serialized_end=241,
)


_HIT = _descriptor.Descriptor(
  name='Hit',
  full_name='Hit',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='docid', full_name='Hit.docid', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tf', full_name='Hit.tf', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=243,
  serialized_end=275,
)


_POSTINGLIST = _descriptor.Descriptor(
  name='PostingList',
  full_name='PostingList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='hits', full_name='PostingList.hits', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=277,
  serialized_end=310,
)

_POSTINGLIST.fields_by_name['hits'].message_type = _HIT
DESCRIPTOR.message_types_by_name['Document'] = _DOCUMENT
DESCRIPTOR.message_types_by_name['Hit'] = _HIT
DESCRIPTOR.message_types_by_name['PostingList'] = _POSTINGLIST

Document = _reflection.GeneratedProtocolMessageType('Document', (_message.Message,), dict(
  DESCRIPTOR = _DOCUMENT,
  __module__ = 'citeomatic.schema_pb2'
  # @@protoc_insertion_point(class_scope:Document)
  ))
_sym_db.RegisterMessage(Document)

Hit = _reflection.GeneratedProtocolMessageType('Hit', (_message.Message,), dict(
  DESCRIPTOR = _HIT,
  __module__ = 'citeomatic.schema_pb2'
  # @@protoc_insertion_point(class_scope:Hit)
  ))
_sym_db.RegisterMessage(Hit)

PostingList = _reflection.GeneratedProtocolMessageType('PostingList', (_message.Message,), dict(
  DESCRIPTOR = _POSTINGLIST,
  __module__ = 'citeomatic.schema_pb2'
  # @@protoc_insertion_point(class_scope:PostingList)
  ))
_sym_db.RegisterMessage(PostingList)


# @@protoc_insertion_point(module_scope)