__version__ = "0.0.5"

import traceback


try:
    from pokemb.mod import PokEmb
except:
    traceback.print_exc()
