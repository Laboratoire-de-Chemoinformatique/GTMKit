GTM Module
==========

The core GTM module provides implementations of Generative Topographic Mapping algorithms.

.. automodule:: gtmkit.gtm
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

BaseGTM
~~~~~~~

.. autoclass:: gtmkit.gtm.BaseGTM
   :members:
   :undoc-members:
   :show-inheritance:

VanillaGTM
~~~~~~~~~~

.. autoclass:: gtmkit.gtm.VanillaGTM
   :members:
   :undoc-members:
   :show-inheritance:

GTM
~~~

.. autoclass:: gtmkit.gtm.GTM
   :members:
   :undoc-members:
   :show-inheritance:

DataStandardizer
~~~~~~~~~~~~~~~

.. autoclass:: gtmkit.gtm.DataStandardizer
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: gtmkit.gtm.squared_euclidean_distance

.. autofunction:: gtmkit.gtm.get_gtm_grid

.. autofunction:: gtmkit.gtm.torch_pca

Usage Examples
--------------

Basic GTM Training
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gtmkit.gtm import GTM
   import torch

   # Generate sample data
   data = torch.randn(1000, 50, dtype=torch.float64)

   # Create and train GTM model
   gtm = GTM(num_nodes=100, num_basis_functions=25)
   latent_coords = gtm.fit_transform(data)

   # Project new data
   responsibilities, log_likelihoods = gtm.project(data)

VanillaGTM vs GTM
~~~~~~~~~~~~~~~~

.. code-block:: python

   from gtmkit.gtm import VanillaGTM, GTM

   # VanillaGTM uses random initialization
   vanilla_gtm = VanillaGTM(num_nodes=100, num_basis_functions=25)

   # GTM uses PCA-based initialization for better convergence
   gtm = GTM(num_nodes=100, num_basis_functions=25)

Data Standardization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gtmkit.gtm import DataStandardizer
   import torch

   # Create standardizer
   standardizer = DataStandardizer(with_mean=True, with_std=True)

   # Fit and transform data
   data = torch.randn(1000, 50)
   standardized_data = standardizer.fit_transform(data)

   # Transform new data using same parameters
   new_data = torch.randn(100, 50)
   standardized_new_data = standardizer.transform(new_data)
