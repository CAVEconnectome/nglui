"""Sharding specification and auto-tuning for precomputed annotations.

Computes optimal neuroglancer_uint64_sharded_v1 parameters from
annotation count and data size.
"""

from typing import Literal, NamedTuple, Optional

MINISHARD_TARGET_COUNT = 1000
SHARD_TARGET_SIZE = 50_000_000  # ~50MB per shard file


class ShardSpec(NamedTuple):
    """Neuroglancer uint64 sharded v1 specification."""

    type: str
    hash: Literal["murmurhash3_x86_128", "identity_hash"]
    preshift_bits: int
    shard_bits: int
    minishard_bits: int
    data_encoding: Literal["raw", "gzip"]
    minishard_index_encoding: Literal["raw", "gzip"]

    def to_json(self) -> dict:
        return {
            "@type": self.type,
            "hash": self.hash,
            "preshift_bits": self.preshift_bits,
            "shard_bits": self.shard_bits,
            "minishard_bits": self.minishard_bits,
            "data_encoding": str(self.data_encoding),
            "minishard_index_encoding": str(self.minishard_index_encoding),
        }


def choose_output_spec(
    total_count: int,
    total_bytes: int,
    hashtype: Literal["murmurhash3_x86_128", "identity_hash"] = "murmurhash3_x86_128",
    gzip_compress: bool = True,
) -> Optional[ShardSpec]:
    """Compute sharding parameters from annotation count and data size.

    Returns None if sharding is not needed (single entry or tensorstore
    unavailable).

    Parameters
    ----------
    total_count : int
        Number of entries.
    total_bytes : int
        Total bytes of encoded data.
    hashtype : str
        Hash function for shard assignment.
    gzip_compress : bool
        Whether to use gzip compression for data and index.

    Returns
    -------
    ShardSpec or None
    """
    if total_count <= 1:
        return None

    total_minishard_bits = 0
    while (total_count >> total_minishard_bits) > MINISHARD_TARGET_COUNT:
        total_minishard_bits += 1

    shard_bits = 0
    while (total_bytes >> shard_bits) > SHARD_TARGET_SIZE:
        shard_bits += 1

    preshift_bits = 0
    while MINISHARD_TARGET_COUNT >> preshift_bits:
        preshift_bits += 1

    minishard_bits = total_minishard_bits - min(total_minishard_bits, shard_bits)

    data_encoding: Literal["raw", "gzip"] = "gzip" if gzip_compress else "raw"
    minishard_index_encoding: Literal["raw", "gzip"] = (
        "gzip" if gzip_compress else "raw"
    )

    return ShardSpec(
        type="neuroglancer_uint64_sharded_v1",
        hash=hashtype,
        preshift_bits=preshift_bits,
        shard_bits=shard_bits,
        minishard_bits=minishard_bits,
        data_encoding=data_encoding,
        minishard_index_encoding=minishard_index_encoding,
    )
