# project dependencies
from promptface.modules.app import app


# --- do something like on/off green LEDs or save data, etc... ---
def on_verify_success(x, y):
    print('x + y = {}'.format(x+y))
    print(app.target_path, app.target_distance)


# --- do something like on/off red LEDs or save data, etc... ---
def on_verify_failure():
    print(app.target_path, app.target_distance)


# How to Use app
app(on_verify_success, on_verify_failure, params1=(1, 3), params2=())

# pass None when you don't want to pass the function in app()
# app(None, None)