import { ID, ModdedDex } from "@pkmn/dex";

import * as fs from "fs";
import path from "path";

function getGenData(gen: number) {
    const format = `gen${gen}` as ID;
    const dex = new ModdedDex(format);
    const data = {
        species: dex.species.all(),
        moves: dex.moves.all(),
        abilities: dex.abilities.all(),
        items: dex.items.all(),
        conditions: dex.data.Conditions,
        typechart: dex.types.all(),
    };
    return data;
}

const data = Object.fromEntries(
    [1, 2, 3, 4, 5, 6, 7, 8, 9].map((gen) => {
        return [`gen${gen}`, getGenData(gen)];
    }),
);

const jsonData: string = JSON.stringify(data, null, 4); // Indented with 4 spaces for readability
const fpath = "data/data.json";

fs.mkdirSync(path.dirname(fpath), { recursive: true });

// Write to a file
fs.writeFile(fpath, jsonData, (err: NodeJS.ErrnoException | null) => {
    if (err) {
        console.error("Error writing to the file:", err);
        return;
    }
    console.log(`JSON data has been written to ${fpath}`);
});
