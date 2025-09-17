import numpy as np
import warnings
import math
from .biot_savart import serial_biot_savart

class Pose:
    """
    Represents the position and orientation of an object in 3D space.
    """
    def __init__(self, position, orientation_axis, orientation_angle):
        """
        Initializes the Pose.

        Args:
            position (np.ndarray): (3,) array for the x, y, z position.
            orientation_axis (np.ndarray): (3,) array for the axis of rotation.
            orientation_angle (float): The angle of rotation in radians.
        """
        self.position = np.array(position)
        self.rotation_matrix = self._rotation_matrix(orientation_axis, orientation_angle)

    def _rotation_matrix(self, axis, angle):
        """
        Generates a rotation matrix.
        """
        axis = axis / np.linalg.norm(axis)
        a = math.cos(angle / 2)
        b, c, d = -axis * math.sin(angle / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, bd, cd = b * c, b * d, c * d
        ad, ac, ab = a * d, a * c, a * b
        return np.array([
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
        ])

class CurrentLoop:
    """
    Base class for a current-carrying loop.
    """
    def __init__(self, current):
        self.current = current
        self.segments = None

    def get_segments(self):
        if self.segments is None:
            raise NotImplementedError("Subclass must implement segment generation.")
        return self.segments

    def biot_savart(self, field_points):
        if self.segments is None:
            raise RuntimeError("Loop segments have not been generated.")
        
        dl_vectors = self.segments[:, 3:6]
        source_points = self.segments[:, 0:3]
        element_lengths = np.linalg.norm(dl_vectors, axis=1)
        element_directions = dl_vectors / element_lengths[:, np.newaxis]

        b_field_vectors = 2 * serial_biot_savart(
            element_centers=source_points,
            element_lengths=element_lengths,
            element_directions=element_directions,
            field_points=field_points
        )
        
        return b_field_vectors * self.current

class RingLoop(CurrentLoop):
    """
    A circular current-carrying loop.
    """
    def __init__(self, current, radius, num_segments, pose):
        super().__init__(current)
        self.radius = radius
        self.num_segments = num_segments
        self.pose = pose
        self._generate_segments()

    def _generate_segments(self):
        segments = np.zeros((self.num_segments, 6))
        d_phi = 2 * np.pi / self.num_segments

        for i in range(self.num_segments):
            phi = i * d_phi
            # Position of the segment in the loop's local frame (xy-plane)
            local_pos = np.array([self.radius * np.cos(phi), self.radius * np.sin(phi), 0])
            # Direction of the segment (tangent)
            local_dir = np.array([-np.sin(phi), np.cos(phi), 0])

            # Rotate and translate to the world frame
            world_pos = self.pose.position + self.pose.rotation_matrix @ local_pos
            world_dir = self.pose.rotation_matrix @ local_dir

            segment_length = self.radius * d_phi
            dl_vector = world_dir * segment_length

            segments[i, 0:3] = world_pos
            segments[i, 3:6] = dl_vector

        self.segments = segments

class RectangularLoop(CurrentLoop):
    """
    A rectangular current-carrying loop.
    """
    def __init__(self, current, width, height, num_segments_per_side, pose):
        super().__init__(current)
        self.width = width
        self.height = height
        self.num_segments_per_side = num_segments_per_side
        self.pose = pose
        self._generate_segments()

    def _generate_segments(self):
        all_segments = []

        # Define corners in local frame
        p1 = np.array([-self.width / 2, -self.height / 2, 0])
        p2 = np.array([self.width / 2, -self.height / 2, 0])
        p3 = np.array([self.width / 2, self.height / 2, 0])
        p4 = np.array([-self.width / 2, self.height / 2, 0])
        corners = [p1, p2, p3, p4]

        for i in range(4):
            start_p = corners[i]
            end_p = corners[(i + 1) % 4]
            side_vector = end_p - start_p
            side_length = np.linalg.norm(side_vector)
            side_direction = side_vector / side_length

            segment_length = side_length / self.num_segments_per_side

            for j in range(self.num_segments_per_side):
                # Position of the segment in the local frame
                local_pos = start_p + side_direction * (j + 0.5) * segment_length

                # Rotate and translate to the world frame
                world_pos = self.pose.position + self.pose.rotation_matrix @ local_pos
                dl_vector = self.pose.rotation_matrix @ (side_direction * segment_length)

                segment_data = np.hstack([world_pos, dl_vector])
                all_segments.append(segment_data)

        self.segments = np.array(all_segments)

class StraightWire(CurrentLoop):
    """
    A straight current-carrying wire.
    """
    def __init__(self, current, start_point, end_point, num_segments):
        super().__init__(current)
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.num_segments = num_segments
        self._generate_segments()

    def _generate_segments(self):
        all_segments = []
        wire_vector = self.end_point - self.start_point
        wire_length = np.linalg.norm(wire_vector)
        wire_direction = wire_vector / wire_length

        segment_length = wire_length / self.num_segments

        for i in range(self.num_segments):
            pos = self.start_point + wire_direction * (i + 0.5) * segment_length
            dl_vector = wire_direction * segment_length
            segment_data = np.hstack([pos, dl_vector])
            all_segments.append(segment_data)

        self.segments = np.array(all_segments)
