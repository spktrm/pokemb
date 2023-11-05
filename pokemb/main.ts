import { ID, ModdedDex } from "@pkmn/dex";
import * as fs from "fs";
import path from "path";

async function getGenData(gen: number) {
    const format = `gen${gen}` as ID;
    const dex = new ModdedDex(format);
    const species = dex.species.all();
    const promises = species.map(species => dex.learnsets.get(species.id));
    const learnsets = await Promise.all(promises);
    const data = {
        species,
        moves: dex.moves.all(),
        abilities: dex.abilities.all(),
        items: dex.items.all(),
        conditions: dex.data.Conditions,
        typechart: dex.types.all(),
        learnsets,
    };
    return data;
}

async function getAllDataAndWriteToFile() {
    const generations = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const dataPromises = generations.map(async (gen) => {
        const genData = await getGenData(gen);
        return [`gen${gen}`, genData];
    });

    const entries = await Promise.all(dataPromises);
    const data = Object.fromEntries(entries);

    const jsonData: string = JSON.stringify(data, null, 4); // Indented with 4 spaces for readability
    const fpath = "pokemb/data/data.json";

    fs.mkdirSync(path.dirname(fpath), { recursive: true });

    // Write to a file
    fs.writeFile(fpath, jsonData, (err: NodeJS.ErrnoException | null) => {
        if (err) {
            console.error("Error writing to the file:", err);
            return;
        }
        console.log(`JSON data has been written to ${fpath}`);
    });
}

// Call the function to perform all operations
getAllDataAndWriteToFile().catch(console.error);
