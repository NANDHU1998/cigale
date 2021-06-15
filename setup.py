from distutils.command.build import build

from setuptools import find_packages, setup


class custom_build(build):
    user_options = [
        ('bc03res=', None, 'Resolution of the BC03 models, hr or lr.'),
        ]
    description = 'Build the pcigale database.'

    def initialize_options(self):
        build.initialize_options(self)
        self.bc03res = 'lr'

    def finalize_options(self):
        assert self.bc03res in ('lr', 'hr'), 'bc03res must be hr or lr!'
        build.finalize_options(self)

    def run(self):
        # Build the database.
        import database_builder
        database_builder.build_base(self.bc03res)

        # Proceed with the build
        build.run(self)

entry_points = {
    'console_scripts': ['pcigale = pcigale:main',
                        'pcigale-plots = pcigale_plots:main',
                        'pcigale-filters = pcigale_filters:main']
}

setup(
    name="pcigale",
    version="dev",
    packages=find_packages(exclude=["database_builder"]),

    install_requires=['numpy', 'scipy', 'matplotlib', 'configobj', 'astropy'],
    setup_requires=['numpy', 'scipy', 'astropy', 'configobj'],
    entry_points=entry_points,

    cmdclass={"build": custom_build},
    package_data={'pcigale': ['data/*/*.pickle',
                              'sed_modules/curves/*.dat'],
                  'pcigale_plots': ['resources/CIGALE.png']},

    include_package_data=True,
    author="The CIGALE team",
    author_email="cigale@lam.fr",
    url="https://cigale.lam.fr",
    description="Python Code Investigating Galaxy Emission",
    license="CeCILL-V2",
    keywords="astrophysics, galaxy, SED fitting"
)
