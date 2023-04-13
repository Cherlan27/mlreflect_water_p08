import numpy as np
from refnx.util import ErrorProp as EP

class Transform_Fresnel(object):
    r"""
    Mathematical transforms of numeric data.

    Parameters
    ----------
    form : None or str
        One of:

            - 'lin'
                No transform is made
            - 'logY'
                log10 transform
            - 'YX4'
                YX**4 transform
            - 'YX2'
                YX**2 transform
            - None
                No transform is made
            - fresnel
                Fresnel reflectivity

    Notes
    -----
    You ask for a transform to be carried out by calling the Transform object
    directly.

    >>> x = np.linspace(0.01, 0.1, 11)
    >>> y = np.linspace(100, 1000, 11)
    >>> y_err = np.sqrt(y)
    >>> t = Transform('logY')
    >>> ty, te = t(x, y, y_err)
    >>> ty
    array([2.        , 2.2787536 , 2.44715803, 2.56820172, 2.66275783,
           2.74036269, 2.80617997, 2.86332286, 2.91381385, 2.95904139,
           3.        ])

    """
    
    def __init__(self, form, **kwargs):
        types = [None, 'lin', 'logY', 'YX4', 'YX2','fresnel']
        self.form = None
        
        if form in types:
            self.form = form
        else:
            raise ValueError("The form parameter must be one of [None, 'lin',"
                             " 'logY', 'YX4', 'YX2', 'fresnel']")
        
        self.qc = kwargs.get('qc')
        self.roughness = kwargs.get('roughness')
    
        if self.form == 'fresnel':
            if self.qc == None:
                raise ValueError("Define critical angle for fresnel reflectivity")
            if self.roughness == None:
                raise ValueError("Define roughness for fresnel reflectivity")                

    def __repr__(self):
        return "Transform({0})".format(repr(self.form))

    def __call__(self, x, y, y_err=None):
        """
        Calculate the transformed data

        Parameters
        ----------
        x : array-like
            x-values
        y : array-like
            y-values
        y_err : array-like
            Uncertainties in `y` (standard deviation)

        Returns
        -------
        yt, et : tuple
            The transformed data

        Examples
        --------
        >>> x = np.linspace(0.01, 0.1, 11)
        >>> y = np.linspace(100, 1000, 11)
        >>> y_err = np.sqrt(y)
        >>> t = Transform('logY')
        >>> ty, te = t(x, y, y_err)
        >>> ty
        array([2.        , 2.2787536 , 2.44715803, 2.56820172, 2.66275783,
               2.74036269, 2.80617997, 2.86332286, 2.91381385, 2.95904139,
               3.        ])

        """
        return self.__transform(x, y, y_err=y_err)

    def __transform(self, x, y, y_err=None):
        r"""
        Transform the data passed in

        Parameters
        ----------
        x : array-like

        y : array-like

        y_err : array-like

        Returns
        -------
        yt, et : tuple
            The transformed data
        """

        if y_err is None:
            etemp = np.ones_like(y)
        else:
            etemp = y_err

        if self.form in ['lin', None]:
            yt = np.copy(y)
            et = np.copy(etemp)
        elif self.form == 'logY':
            yt, et = EP.EPlog10(y, etemp)
        elif self.form == 'YX4':
            yt = y * np.power(x, 4)
            et = etemp * np.power(x, 4)
        elif self.form == 'YX2':
            yt = y * np.power(x, 2)
            et = etemp * np.power(x, 2)
        elif self.form == 'fresnel':
            y_2 = y / (np.exp(-x**2 * self.roughness**2) * abs((x-np.sqrt((x**2 - self.qc**2)+0j))/(x+np.sqrt((x**2 - self.qc**2)+0j)))**2)
            e_2 = etemp / (np.exp(-x**2 * self.roughness**2) * abs((x-np.sqrt((x**2 - self.qc**2)+0j))/(x+np.sqrt((x**2 - self.qc**2)+0j)))**2)
            # y_2 = abs(np.array(y_2))
            # print(y_2)
            yt, et = EP.EPlog10(y_2, e_2)
        if y_err is None:
            return yt, None
        else:
            return yt, et
