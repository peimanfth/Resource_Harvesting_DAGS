const composer = require("openwhisk-composer");

module.exports = composer.sequence(
  composer.action("partitioner"),
  composer.parallel(
    composer.action("mapper0"),
    composer.action("mapper1"),
    composer.action("mapper2")
  ),
  composer.action("reducer")
);
