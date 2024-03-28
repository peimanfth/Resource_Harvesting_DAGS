const composer = require("openwhisk-composer");

module.exports = composer.sequence(
  composer.action("wait1_B"),
  composer.parallel(
    composer.action("AES1_B"),
    composer.action("AES2_B"),
    composer.action("AES3_B")
  ),
  composer.action("Stats_B")
);
