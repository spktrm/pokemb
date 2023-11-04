__version__ = "0.0.4"

import traceback


try:
    from pokemb.mod import PokEmb
except:
    traceback.print_exc()
