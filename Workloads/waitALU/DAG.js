const composer = require("openwhisk-composer");
module.exports = composer.sequence(
  composer.action("waitInput"),
  composer.action("ALU")
);
