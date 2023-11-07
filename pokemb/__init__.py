__version__ = "0.0.10"

import traceback


try:
    from pokemb.mod import PokEmb
except:
    traceback.print_exc()
