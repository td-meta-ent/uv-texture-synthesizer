# Copyright (c) 2023 Metaverse Entertainment Inc. - All Rights Reserved.

from typing import List, Tuple

import cv2
import numpy as np
from nptyping import Float32, NDArray, Shape, UInt8


class BilinearInterpolator:

    def __init__(self, image: NDArray[Shape['*, *, 3'], UInt8]):
        """
        Initializes the BilinearInterpolator instance with a given image.

        Args:
            image: A NumPy array representing the image.
        """
        self.image = image
        self.image_height, self.image_width = self.image.shape[:2]

    def bilinear_interpolation_at_point(
            self, point: Tuple[float,
                               float]) -> NDArray[Shape['*, 3'], Float32]:
        """
        Performs bilinear interpolation at a given point in the image.

        Args:
            point: The x and y coordinates of the point in the image.

        Returns:
            The color value at the given point after bilinear interpolation.
        """

        # Compute the interpolation parameters
        direction, center_coordinates, weights = self._compute_interpolation_parameters(
            point)

        # Sample colors at the four corners around the point
        color_samples = self._sample_colors_around_point(
            center_coordinates, direction)

        # Perform bilinear interpolation using the sampled colors and weights
        interpolated_color = self._interpolate_color_samples(
            color_samples, weights)

        return interpolated_color

    def _sample_colors_around_point(
        self, center_coordinates: Tuple[int, int], direction: List[str]
    ) -> Tuple[NDArray[Shape['*, 3'], UInt8], NDArray[Shape['*, 3'], UInt8],
               NDArray[Shape['*, 3'], UInt8], NDArray[Shape['*, 3'], UInt8]]:
        """
        Samples the colors at the four corners around a central point.

        Args:
            center_coordinates: The integer part of the center coordinates.
            direction: The direction of interpolation.

        Returns:
            A tuple containing the sampled colors.
        """
        x, y = center_coordinates
        color_a: NDArray[Shape['3'], UInt8] = self.image[y, x]
        color_b: NDArray[Shape['3'], UInt8] = self.image[
            y, x - 1] if direction[0] == 'left' else self.image[y, x + 1]
        color_c: NDArray[Shape['3'], UInt8] = self.image[
            y - 1, x] if direction[1] == 'up' else self.image[y + 1, x]
        color_d: NDArray[Shape['3'], UInt8] = \
            self.image[y - 1, x - 1] if direction[0] == 'left' and direction[1] == 'up' else \
            self.image[y + 1, x - 1] if direction[0] == 'left' and direction[1] == 'down' else \
            self.image[y - 1, x + 1] if direction[0] == 'right' and direction[1] == 'up' else \
            self.image[y + 1, x + 1]

        return color_a, color_b, color_c, color_d

    def _interpolate_color_samples(
        self, color_samples: Tuple[NDArray[Shape['*, 3'], UInt8],
                                   NDArray[Shape['*, 3'],
                                           UInt8], NDArray[Shape['*, 3'],
                                                           UInt8],
                                   NDArray[Shape['*, 3'],
                                           UInt8]], weights: Tuple[float, float,
                                                                   float, float]
    ) -> NDArray[Shape['*, 3'], Float32]:
        """Interpolates the sampled colors using the specified weights.

        Args:
            color_samples: The sampled colors at the four corners.
            weights: The weights for interpolation.

        Returns:
            The interpolated color.
        """
        return color_samples[0] * weights[0] + \
            color_samples[1] * weights[1] + \
            color_samples[2] * weights[2] + \
            color_samples[3] * weights[3]

    def _compute_interpolation_parameters(
        self, center_point: Tuple[float, float]
    ) -> Tuple[List[str], Tuple[int, int], Tuple[float, float, float, float]]:
        """Computes the parameters required for bilinear interpolation.

        Args:
            center_point: The x and y coordinates of the center point.

        Returns:
            A tuple containing the direction of interpolation, the integer part of the center coordinates,
            the fractional part of the center coordinates, and the weights for interpolation.
        """
        direction = ['right', 'down']
        center_x, center_y = int(center_point[0]), int(center_point[1])
        frac_x, frac_y = center_point[0] - center_x, center_point[1] - center_y

        if frac_x < 0:
            direction[0] = 'left'
            frac_x = -frac_x
        if frac_y < 0:
            direction[1] = 'up'
            frac_y = -frac_y

        x, y, r, s = frac_x, 1 - frac_x, frac_y, 1 - frac_y
        area_a, area_b, area_c, area_d = x * r, y * r, x * s, y * s
        area_sum = area_a + area_b + area_c + area_d

        weight_a, weight_b, weight_c, weight_d = area_d / area_sum, area_c / area_sum, area_b / area_sum, area_a / area_sum

        return direction, (center_x, center_y), (weight_a, weight_b, weight_c,
                                                 weight_d)
