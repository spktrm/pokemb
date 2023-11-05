data:
	tsc
	node dist/main.js
	python scripts/process.py

clean:
	rm -rf pokemb/data