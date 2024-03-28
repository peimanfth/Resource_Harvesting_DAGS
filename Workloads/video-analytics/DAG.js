const composer = require("openwhisk-composer");

module.exports = composer.sequence(
  composer.action("streaming"),
  composer.action("decoder"),
  composer.parallel(
    composer.action("recognition1"),
    composer.action("recognition2")
  )
);
