__version__ = "0.0.8"

import traceback


try:
    from pokemb.mod import PokEmb
except:
    traceback.print_exc()
