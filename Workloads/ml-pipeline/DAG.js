const composer = require("openwhisk-composer");

module.exports = composer.sequence(
  composer.action("pca"),
  composer.parallel(
    composer.action("paramtune1"),
    composer.action("paramtune2"),
    composer.action("paramtune3"),
    composer.action("paramtune4")
  ),
  composer.action("combine")
);
