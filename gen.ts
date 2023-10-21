import { TeamValidator } from "@pkmn/sim";

import * as fs from "fs";

const format = "gen9ou";
const validator = TeamValidator.get(format);
const dex = validator.dex;

const data = {
    species: dex.species.all(),
    moves: dex.moves.all(),
    abilities: dex.abilities.all(),
    items: dex.items.all(),
    conditions: dex.data.Conditions,
};

const jsonData: string = JSON.stringify(data, null, 4); // Indented with 4 spaces for readability

// Write to a file
fs.writeFile("data.json", jsonData, (err: NodeJS.ErrnoException | null) => {
    if (err) {
        console.error("Error writing to the file:", err);
        return;
    }
    console.log("JSON data has been written to data.json");
});
