import * as fs from "fs";
import path from "path";
import {
    Ability,
    ID,
    Item,
    Learnset,
    ModdedDex,
    Move,
    Species,
    Type,
} from "@pkmn/dex";

function findDuplicates(arr: string[]): string[] {
    return arr.filter((item, index) => {
        return arr.indexOf(item) !== index;
    });
}

// Helper function to create an enumeration from an array
function enumerate(arr: string[]): { [key: string]: number } {
    const enumeration: { [key: string]: number } = {};
    const dupes = findDuplicates(arr);
    arr.forEach((item, index) => {
        enumeration[item] = index;
    });
    return enumeration;
}

type GenData = {
    species: Species[];
    moves: Move[];
    abilities: Ability[];
    items: Item[];
    typechart: Type[];
    learnsets: Learnset[];
};

function mapId<T extends { id: string; [key: string]: any }>(
    arr: T[],
): string[] {
    return arr.map((item) => item.id);
}

function getMoveId(item: { [k: string]: any }): string {
    if (item.id === "hiddenpower") {
        const moveType = item.type.toLowerCase();
        if (moveType === "normal") {
            return item.id;
        } else {
            return `${item.id}${moveType}`;
        }
    } else if (item.id === "return") {
        return `${item.id}102`;
    } else {
        return item.id;
    }
}

function formatData(data: GenData) {
    const moveIds = [...data.moves.map((item) => getMoveId(item)), "return"];
    return {
        species: enumerate(mapId(data.species)),
        moves: enumerate(moveIds),
        abilities: enumerate(mapId(data.abilities)),
        items: enumerate(mapId(data.items)),
    };
}

async function getGenData(gen: number) {
    const format = `gen${gen}` as ID;
    const dex = new ModdedDex(format);
    const species = dex.species.all();
    const promises = species.map((species) => dex.learnsets.get(species.id));
    const learnsets = await Promise.all(promises);
    const data = {
        species: [...species],
        moves: [...dex.moves.all()],
        abilities: [...dex.abilities.all()],
        items: [...dex.items.all()],
        typechart: [...dex.types.all()],
        learnsets: [...learnsets],
    };
    return data;
}

function addIndices(
    gendata: {
        species: Species[];
        moves: Move[];
        abilities: Ability[];
        items: Item[];
        typechart: Type[];
        learnsets: Learnset[];
    },
    indices: {
        species: { [key: string]: number };
        moves: { [key: string]: number };
        abilities: { [key: string]: number };
        items: { [key: string]: number };
    },
) {
    return {
        ...gendata,
        species: gendata.species.map((x) => {
            return { ...x, index: indices.species[x.id] };
        }),
        moves: gendata.moves.map((x) => {
            return { ...x, index: indices.moves[getMoveId(x)] };
        }),
        abilities: gendata.abilities.map((x) => {
            return { ...x, index: indices.abilities[x.id] };
        }),
        items: gendata.items.map((x) => {
            return { ...x, index: indices.items[x.id] };
        }),
        typechart: gendata.typechart,
        learnsets: gendata.learnsets,
    };
}

async function getAllDataAndWriteToFile() {
    const generations = [1, 2, 3, 4, 5, 6, 7, 8, 9];

    const gen9data = formatData(await getGenData(9));
    const dataPromises = generations.map(async (gen) => {
        const genData = addIndices(await getGenData(gen), gen9data);
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
