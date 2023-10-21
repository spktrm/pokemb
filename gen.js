"use strict";
exports.__esModule = true;
var sim_1 = require("@pkmn/sim");
var fs = require("fs");
var format = "gen9ou";
var validator = sim_1.TeamValidator.get(format);
var dex = validator.dex;
var data = {
    species: dex.species.all(),
    moves: dex.moves.all(),
    abilities: dex.abilities.all(),
    items: dex.items.all(),
    conditions: dex.data.Conditions
};
var jsonData = JSON.stringify(data, null, 4); // Indented with 4 spaces for readability
// Write to a file
fs.writeFile("data.json", jsonData, function (err) {
    if (err) {
        console.error("Error writing to the file:", err);
        return;
    }
    console.log("JSON data has been written to data.json");
});
