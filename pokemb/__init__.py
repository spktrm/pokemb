__version__ = "0.1.0"

import traceback


try:
    from pokemb.mod import PokEmb
except:
    traceback.print_exc()
