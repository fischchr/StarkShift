import re
import numpy as np
import logging


def _check_all_characters(string: str) -> set:
    """Check that all characters in `string` are valid. 
    
    # Arguments
    * string::str - Evaluation string

    # Returns
    * diff::set - Set of characters which are in `string` but not in `allowed_words`. 
                  If `diff` is empty, all characters in `string` are valid.
    """

    logging.debug('_check_all_characters')

    # Define the allowed words
    allowed_words = ['x', 'y', 'z', 'sin', 'cos', 'sqrt', 'pi', 'i', 'j', '+', '-', '/', '(', ')', '.'] + [str(i) for i in range(10)]
    # Turn them into a list of characters
    allowed_characters = [c for word in allowed_words for c in word]
    # Turn the allowed characters into a set (remove doubles)
    allowed_characters = set(allowed_characters)
    logging.debug(f'Defining allowed characters: {allowed_characters}')

    # Turn the string into a set of letters
    characters =  [c for c in string]
    characters = set(characters)
    logging.debug(f'Getting characters from string <{string}>: {characters}')

    # Check whether a character in characters is not in allowed_characters
    diff = characters.difference(allowed_characters)
    logging.debug(f'Difference of the two sets: {diff}')

    # Return whether the difference is empty (all characters in string are in allowed_characters)
    return diff


def _preformat_string(string: str) -> str:
    """Preformat a vectro description string.
    Remove everything that can cause errors in the evaluation of the string.
    
    # Arguments
    * `string` - Evaluation string

    # Returns
    * `formatted_string` - String with all '*' and 'np.' removed and 'i' replaced by 'j'
    """

    logging.debug('_preformat_string')
    logging.debug(f'received string: <{string}>')

    # Remove all white spaces
    string = re.sub(' ', '', string)
    # Remove all *
    string = re.sub('\*', '', string)
    # Replace all 'i' by 'j' 
    string = re.sub('i', 'j', string)
    # Correct for mistakes introduced in the previous step
    string = re.sub('pj', 'pi', string) 
    string = re.sub('sjn', 'sin', string)
    # Remove all np.
    string = re.sub('np.', '', string)

    # Log the result
    logging.debug(f'preformatted string: <{string}>')

    return string


def _mark_math_expressions(string: str) -> str:
    """Enclose all mathematical expressions with a pair of $ and add np.* to all mathematical functions.

    # Arguments
    * `string` - Evaluation string

    # Returns
    * `formatted_string` - String with sin, cos, sqrt and pi marked by $.
    """

    logging.debug('_mark_math_expressions')

    # Replace all math expressions with numpy function calls
    string = re.sub('sqrt', '$np.sqrt$', string)
    string = re.sub('sin', '$np.sin$', string)
    string = re.sub('cos', '$np.cos$', string)
    string = re.sub('pi', '$np.pi$', string) 

    # Log the result
    logging.debug(f'Marked string: <{string}>')

    return string


def _format_eval_string(string: str) -> str:
    """Format the evaluation string for evaluation. """

    logging.debug('_format_eval_string')

    # Find all 

    # Get the first letter of the string. 
    eval_string = string[0] if string[0] != 'j' else '1j'
    i = 1

    # Make sure the string does not start with a math expression
    if eval_string[0] == '$':
        # Find the math expression
        s = re.search('\$.*?\$', string)
        # Log the result
        logging.debug(f'Found math expression: <{s.group()}>')
        # Put in the math expression
        eval_string = s.group()
        # Skip ahead to the end of the math expression
        i = s.end()
   
    while i < len(string):
        # Log the current eval_string value
        logging.debug(f'Current eval_string: <{eval_string}>, index: {i}')

        # Get the i-th character
        c = string[i]

        # Format math expressions
        if c == '$':
            # Insert a multiplication operator if needed
            if not (eval_string[-1] in ['+', '-', '/', '(']):
                eval_string += '*'
                logging.debug('Inserted multiplication operator')
            # Find the math expression
            s = re.search('\$.*?\$', string[i:]).group()
            # Log the result
            logging.debug(f'Found math expression: <{s}>')
            # Add it
            eval_string += s
            # Skip ahead to the end of the math expression
            i += len(s)
            logging.debug(f'Skip to character {string[i]}, index: {i}')

            # Go to the next character
            continue
        
        # Format everything else
        if (c in ['x', 'y', 'z', '(', '$']) and (not (eval_string[-1] in ['+', '-', '/', '(', '$'])): 
            # If c is either 'x', 'y', 'z', an opening parenthesis or a math operator
            # And the preceeding character was not a mathematical operator or expression
            # a multiplication operator has to be inserted
            eval_string += '*' + c
            # Log
            logging.debug('Inserted multiplication operator')
        elif c == 'j': 
            # If we have a j with no number in front of it put one in
            if eval_string[-1] in  ['.'] + [str(i) for i in range(10)]:
                eval_string += 'j'
            else:
                eval_string += '1j'
        else:
            # Otherwise just add the current character 
            eval_string += c
        
        # Continue with the next letter
        i += 1

    logging.debug(f'Final eval string: <{eval_string}>')

    return eval_string


def _remove_math_marking(string: str) -> str:
    """Remove all dollar signs. """

    return re.sub('\$', '', string)


def evaluate_vector_description(string: str) -> np.ndarray:
    """Evaluate a string containing a mathematical description of a vector.

    The evaluation is done in the following way
    1. Remove all characters that could interfere with the following steps.
       (Whitespaces, *, np.). Also replace all i by j (imaginary unit).
    2. Check that there are no invalid characters.
    3. Surround all math expressions (sin, cos, sqrt, pi) by a pair of dollar signs such that they can be evaluated individually.
    4. Insert multiplication signs where necessary
    5. Evaluate the string using numpy

    # Arguments
    * string::str - (Non-normalized) description of a vector.
                    Examples:
                        'z'
                        '2x + iy - z'
                        '-1/sqrt(2) * (x + iy)'

    # Returns
    * epsilon::array(3) - The normalized vector described by the string in carthesian coordinates. 
                          For the examples from above:
                            [0, 0, 1]
                            [0.81649658+0.j, 0.+0.40824829j, -0.40824829+0.j]
                            [-0.70710678+0.j, -0.-0.70710678j, 0.+0.j]
    """

    logging.debug('evaluate_vector_description')
    logging.debug(f'Received string: <{string}>')

    # Get basis vectors
    x, y, z = np.eye(3)

    # Preformat string
    string = _preformat_string(string)
    # Make sure there is no invalid input
    invalid_chars = _check_all_characters(string)

    if invalid_chars != set():
        raise ValueError(f'Invalid characters {invalid_chars}')

    # Mark math expressions
    string = _mark_math_expressions(string)

    # Format the evaluation string
    eval_string = _format_eval_string(string)

    # Remove the math markers again        
    eval_string = _remove_math_marking(eval_string)
    logging.debug(f'Final evaluation string: <{eval_string}>')

    # Evaluate expression
    epsilon = eval(eval_string)
    
    # Evaluate expression
    return epsilon / np.sqrt(np.sum(epsilon*epsilon.conj()))


def cart_to_sph(vec_xyz: np.ndarray) -> np.ndarray:
    """Convert a vector in cartesian basis into a vector in spherical basis.

    Cartesian basis:
        x = [1, 0, 0]
        y = [0, 1, 0]
        z = [0, 0, 1]

    Spherical basis:
        e_- = [1, 0, 0]
        e_0 = [0, 1, 0]
        e_+ = [0, 0, 1]
    
    # Arguments
    * vec_xyz::array(3) - Vector in cartesian basis. Shape = (3,)

    # Returns
    * vec_sph::array(3) - Vector in spherical basis. Shape = (3,)
    """

    # Calculate the spherical coordinates
    Am = (vec_xyz[0] - 1j * vec_xyz[1]) / np.sqrt(2)
    A0 = vec_xyz[2]
    Ap = - (vec_xyz[0] + 1j * vec_xyz[1]) / np.sqrt(2)

    # make a spherical vector
    vec_sph = make_sph_vector(Am, A0, Ap)

    # Turn them into a spherical vector (minus sign for components -1 and 1)
    return np.real_if_close(vec_sph)


def sph_to_cart(vec_sph: np.ndarray) -> np.ndarray:
    """Convert a vector in sperical basis into a vector in cartesian basis.

    Cartesian basis:
        x = [1, 0, 0]
        y = [0, 1, 0]
        z = [0, 0, 1]

    Spherical basis:
        e_- = [1, 0, 0]
        e_0 = [0, 1, 0]
        e_+ = [0, 0, 1]
    
    # Arguments
    * vec_sph::array(3) - Vector in sperical basis (-A_-, A_0, -A_p). Shape = (3,)

    # Returns
    * vec_xyz::array(3) - Vector in cartesian basis. Shape = (3,)
    """

    # Get the spherical coordinates from the vector
    Am, A0, Ap = get_sph_components(vec_sph)

    # Calculate the cartesian coordinates
    Ax = (Am - Ap) / np.sqrt(2)
    Ay = 1j * (Am + Ap) / np.sqrt(2)
    Az = A0

    return np.array([Ax, Ay, Az])


def get_sph_components(vec_sph: np.ndarray) -> tuple:
    """Get the components A_-, A_0, and A_+ of a spherical vector. 

    Spherical basis:
        e_- = [1, 0, 0]
        e_0 = [0, 1, 0]
        e_+ = [0, 0, 1]
    
    # Arguments
    * vec_sph::array(3) - Vector in sperical basis (-A_+, A_0, -A_-). Shape = (3,)

    # Returns
    * (A_-, A_0, A_+)::tuple - Components of the spherical vector, s.t.
                               vec_sph = (-A_+ * e_-) + (A_0 * e_0) + (-A_- * e_+)
    """

    # Get the spherical coordinates from the vector
    Am = -vec_sph[2] # Component that is multiplied with e_+
    A0 = vec_sph[1]  # Component that is multiplied with e_0
    Ap = -vec_sph[0] # Component that is multiplied with e_-

    return (Am, A0, Ap)


def make_sph_vector(Am: np.complex, A0: np.complex, Ap: np.complex) -> np.ndarray:
    """Get a spherical vector the components A_-, A_0, and A_+ of a spherical vector. 
       vec_sph = (-A_+ * e_-) + (A_0 * e_0) + (-A_- * e_+)

    Spherical basis:
        e_- = [1, 0, 0]
        e_0 = [0, 1, 0]
        e_+ = [0, 0, 1]
    
    # Arguments
    * (A_-, A_0, A_+)::complex - Components of the spherical vector.

    # Returns
    * vec_sph::array(3) - Vector in sperical basis (-A_+, A_0, -A_-). Shape = (3,)
    """

    return np.array([-Ap, A0, -Am])



if __name__ == '__main__':
    pass