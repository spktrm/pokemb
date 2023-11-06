__version__ = "0.0.9"

import traceback


try:
    from pokemb.mod import PokEmb
except:
    traceback.print_exc()
