import composer
def main():
    return composer.sequence(
        composer.action('hi'),
        composer.action('bye')
    )

