const composer = require("openwhisk-composer");

module.exports = composer.sequence(
  composer.action("wait1_C"),
  composer.parallel(
    composer.action("AES1_C"),
    composer.action("AES2_C"),
    composer.action("AES3_C")
  ),
  composer.action("Stats_C")
);
