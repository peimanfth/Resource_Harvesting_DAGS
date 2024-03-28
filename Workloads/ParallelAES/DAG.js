const composer = require("openwhisk-composer");

module.exports = composer.sequence(
  composer.action("wait1"),
  composer.parallel(
    composer.action("AES1"),
    composer.action("AES2"),
    composer.action("AES3")
  ),
  composer.action("Stats")
);
