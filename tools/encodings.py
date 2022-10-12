from six import PY2, PY3


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, (int, float)):
        text = '{}'.format(text)
    if PY3:
        if isinstance(text, str):  # py3-str is unicode
            return text
        elif isinstance(text, bytes):  # py3-bytes is py2-str
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif PY2:
        if isinstance(text, str):  # py2-str is py3-bytes
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):  # py2-unicode is py3-str
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_to_bytes(text):
    if PY2 and isinstance(text, str):
        return text
    elif PY3 and isinstance(text, bytes):
        return text
    u_text = convert_to_unicode(text)
    return u_text.encode('utf-8')


def recursive_encoding_unification(cur_node):
    from collections import OrderedDict
    reu = recursive_encoding_unification

    if isinstance(cur_node, (list, tuple)):
        return type(cur_node)(
            [reu(item) for item in cur_node])
    elif isinstance(cur_node, (dict, OrderedDict)):
        return type(cur_node)(
            [(reu(k), reu(v)) for (k, v) in cur_node.items()])
    elif isinstance(cur_node, (int, float)):
        return cur_node
    elif cur_node is None:
        return None
    else:  # str, bytes, unicode
        # only convert leaf-nodes
        return convert_to_unicode(cur_node)