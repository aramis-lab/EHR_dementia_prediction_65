import datetime

def check_valid_input(country, age, height_range, weight_range, date_str, min_before, min_during, min_after):    
    """
    Checks that the input values are valid.
    Raises:
        - AssertionError: If any of the following conditions are not met:
            - country is not equal to either 'UK' or 'FR'.
            - age is not equal to either 65 or 70.
            - height_range is not a tuple of 2 integers.
            - height_range is not ordered from smallest to largest.
            - weight_range is not a tuple of 2 positive floats.
            - weight_range is not ordered from smallest to largest.
        - ValueError: If the date_str is not in the format yyyy-mm-dd.
    """    
    assert country in {'UK', 'FR'}, \
    f"country` must be a string equal to either 'UK' or 'FR' not {country}."
    assert age in {65, 70}, \
    f"`age` must be an int equal to either 65 or 70, not {age}."
    
    if height_range:
        assert len(height_range) == 2, f"`height_range` must be a tuple of int of size 2, not {len(height_range)}."
        assert height_range[0] < height_range[1], f"`height_range` should be ordered"
    if weight_range:
        assert len(weight_range) == 2, f"`weight_range` must be a tuple of float of size 2, not {len(weight_range)}."
        assert weight_range[0] > 0 and weight_range[1] > 0, "`weight_range` must be a tuple of positive float"
        assert weight_range[0] < weight_range[1], f"`weight_range` should be ordered"
    
    try:
        datetime.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect date format, should be yyyy-mm-dd")
    
    assert (min_before >=1 or min_during>=1) and  (min_after>=1), "either `min_after` and either`min_during` or `min_before`  should be integers greater than 1 (assumption used in the function get_status)"