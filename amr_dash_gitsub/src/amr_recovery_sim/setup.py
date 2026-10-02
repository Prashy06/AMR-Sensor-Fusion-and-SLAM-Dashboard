from setuptools import find_packages, setup

package_name = 'amr_recovery_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orin',
    maintainer_email='rohitpranauv@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'network_sim = amr_recovery_sim.network_sim:main',
            'fleet_manager = amr_recovery_sim.fleet_manager:main',
            'amr_agent = amr_recovery_sim.amr_agent:main',
            'hitl_bridge = amr_recovery_sim.hitl_bridge:main',
        ],
    },
)
