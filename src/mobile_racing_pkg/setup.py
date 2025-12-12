from setuptools import setup

package_name = 'mobile_racing_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zlyoon',
    maintainer_email='zlyoon@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kanayama_controller = mobile_racing_pkg.kanayama_controller_node_2:main',
            'kanayama_controller_3 = mobile_racing_pkg.kanayama_controller_node3_2:main',
            'kanayama_controller_4 = mobile_racing_pkg.kanayama_controller_node4_2:main',
            #'kanayama_controller = mobile_racing_pkg.kanayama_controller_node:main',
            #'pid_controller = mobile_racing_pkg.pid_controller_node:main',
            #'pure_pursuit_controller = mobile_racing_pkg.pure_pursuit_controller_node:main',
        ],
    },
)
