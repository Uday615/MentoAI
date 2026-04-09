import sys
import traceback

try:
    import backend.main
    print('Success')
except Exception as e:
    print(traceback.format_exc())
    sys.exit(1)
