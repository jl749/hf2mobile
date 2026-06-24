

def check_parent_field(obj, field_name):
    if not hasattr(obj, field_name):
        raise AttributeError(f"{obj.__class__.__name__} is missing required field `{field_name}`. Please make sure you set `self.{field_name}` before calling ABC initialization.")

