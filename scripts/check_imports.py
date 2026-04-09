import importlib
import runpy

try:
    m1 = importlib.import_module('pymongo')
    m2 = importlib.import_module('passlib')
    print('pymongo', getattr(m1, '__version__', 'unknown'))
    print('passlib', getattr(m2, '__version__', 'unknown'))
except Exception as e:
    print('IMPORT_ERROR', e)

print('\nRunning test_main.py')
try:
    runpy.run_path('test_main.py', run_name='__main__')
except SystemExit as se:
    print('test_main exited with', se.code)
except Exception as e:
    import traceback
    traceback.print_exc()
