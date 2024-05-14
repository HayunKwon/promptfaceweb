# project dependencies
from promptface import Promptface


# --- do something like on/off green LEDs or save data, etc... ---
def on_verify_success(x, y):
    print('x + y = {}'.format(x+y))
    print(Promptface.app.target_path, Promptface.app.target_distance)


# --- do something like on/off red LEDs or save data, etc... ---
def on_verify_failure():
    print(Promptface.app.target_path, Promptface.app.target_distance)


# How to Use app
Promptface.app(on_verify_success, on_verify_failure, params1=(1, 3), params2=())

# pass None when you don't want to pass the function in app()
# app(None, None)