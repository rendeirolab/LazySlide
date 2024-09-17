{{ fullname | escape | underline }}


.. currentmodule:: {{ module }}


{% if objtype in ['class'] %}
.. auto{{ objtype }}:: {{ objname }}
    :show-inheritance:
    :special-members: __call__

{% else %}
.. auto{{ objtype }}:: {{ objname }}

{% endif %}