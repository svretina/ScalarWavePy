#!/usr/bin/env python3

import numpy as np


class SingleDomain:
    def __init__(self, physical_domain):
        """Constructor of :py:class:`~.SingleDomain`.
        :param physical_domain: a list of the physical domain
        eg [0,1]
        :type physical_domain: list or np.ndarray
        """
        if isinstance(physical_domain, list):
            self.domain = np.array(physical_domain)
        elif isinstance(physical_domain, np.ndarray):
            self.domain = physical_domain
        else:
            raise TypeError(
                f"Pass list or numpy ndarray.\
                You passed {type(physical_domain)}"
            )


class MultipleDomain:
    def __init__(self, domain_list):
        """Constructor of Domain class
        :param npoints: List of the domains,
        for example for a domain
        x\in[0,1]U[3,4] = np.array([0,1],[3,4])
        :type domain_list: np.ndarray
        """
        if not isinstance(domain_list, np.ndarray):
            domain_list = np.asarray(domain_list).reshape((-1, 2))
        else:
            domain_list = domain_list.reshape((-1, 2))
        self.domain_list = domain_list
        self.ndomains = domain_list.shape[0]
        sdomain_collection = np.empty(self.ndomains, dtype=object)
        for i in range(self.ndomains):
            sdomain_collection[i] = SingleDomain(domain_list[i, :])
        self.domain = sdomain_collection
