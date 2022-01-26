class MultipleDomain:
    def __init__(self, domain_list):
        """Constructor of Domain class
        :param npoints: List of the domains,
        for example for a domain
        x\in[0,1]U[3,4] = np.array([0,1],[3,4])
        :type npoints: np.ndarray
        """
        if not isinstance(xlist, np.ndarray):
            xlist = np.array(xlist)
        assert xlist.shape[1] == 2
        self.domain_list = xlist
        # self.mins = np.apply_along_axis(np.min, 1, xlist)
        # self.maxs = np.apply_along_axis(np.max, 1, xlist)
        # self.oamin = np.min(xlist)
        # self.oamax = np.max(xlist)


class Grid(BaseNumerical):
    def __init__(self, domain, npoints):
        if not isinstance(npoints, np.ndarray):
            npoints = np.array(npoints)
        assert domain.domain_list.shape[0] == npoints.shape[0]
        self.domain = domain

        self.npoints = npoints
        self.n = npoints
        self.ugrids, self.dx = self.get_grids()
        assert self.dx.shape == self.npoints.shape

    def get_grids(self):
        num_subdomains = self.domain.domain_list.shape[0]
        dxs = np.empty((num_subdomains))
        grid = np.empty((num_subdomains), dtype="object")

        for i in range(num_subdomains):
            subdomain = self.domain.domain_list[i, :]
            grid[i] = UniformGrid(subdomain, self.npoints[i])
            dxs[i] = grid[i].dx
        return grid, dxs

    @property
    def shape(self):
        return (self.npoints.shape[0], tuple(self.npoints))

    def _apply_reduction(self, reduction, *args, **kwargs):
        """Apply a reduction to the data.
        :param function: Function to apply to the data.
        :type function: callable
        :return: Reduction applied to the data.
        :rtype: float
        """
        return reduction(self.ugrids, *args, **kwargs)

    def _apply_unary(self, function, *args, **kwargs):
        """Apply a unary function to the data.
        :param function: Unary function.
        :type function:  callable
        :returns: Function applied to the data.
        :rtype:    :py:class:`~.UniformGrid`
        """
        return function(self.ugrids, *args, **kwargs)

    def _apply_binary(self, other, function, *args, **kwargs):
        """This is an abstract function that is used to implement mathematical
        operations with other series (if they have the same x) or
        scalars.
        :py:meth:`~._apply_binary` takes another object that can be of the same
        type or a scalar, and applies ``function(self.y, other.y)``, performing type
        checking.
        :param other: Other object.
        :type other: :py:class:`~.BaseSeries` or derived class or float
        :param function: Dyadic function (function that takes two arguments).
        :type function: callable
        :returns:  Return value of ``function`` when called with self and other.
        :rtype:   :py:class:`~.BaseSeries` or derived class (typically)
        """
        # If it is a number
        if isinstance(other, (int, float, complex)):
            return (function(self.ugrids, other, *args, **kwargs),)

        # If we are here, it is because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")
