pub type Address = u64;
pub type Size = usize;
pub type Symbol = &'static str;
pub use crate::decompile::passes::csh_pass::*;
use crate::mreg::Mreg;
use crate::x86::asm::{Ireg, TestCond};
use crate::x86::op::{Addressing, Comparison, Condition, Operation, Ptrofs, F32, F64};
use either::Either;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use strum_macros::EnumString;
pub type Ident = usize;

pub type MregArgs = Arc<Vec<Mreg>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamType {
    Pointer,
    StructPointer(usize),
    Typed(XType),
    Integer,
    #[allow(dead_code)]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Displacement {
    Const(i64),
    Symbol { ident: Ident, ofs: i64 },
}

impl From<i64> for Displacement {
    fn from(d: i64) -> Self {
        Displacement::Const(d)
    }
}

impl From<(Ident, i64)> for Displacement {
    fn from((ident, ofs): (Ident, i64)) -> Self {
        Displacement::Symbol { ident, ofs }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Addrmode {
    pub base: Option<Ireg>,
    pub index: Option<(Ireg, i64)>,
    pub disp: Displacement,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MachInst {
    Mgetstack(i64, Typ, Mreg),
    Msetstack(Mreg, i64, Typ),
    #[allow(dead_code)]
    Mgetparam(i64, Typ, Mreg),
    Mreturn,
    Mgoto(Symbol),
    Mtailcall(Either<Mreg, Either<Symbol, i64>>),
    Mcall(Either<Mreg, Either<Symbol, i64>>),
    Mstore(MemoryChunk, Addressing, MregArgs, Mreg),
    Mload(MemoryChunk, Addressing, MregArgs, Mreg),
    Mcond(Condition, MregArgs, Symbol),
    Mop(Operation, MregArgs, Mreg),
    Mbuiltin(String, Vec<BuiltinArg<Mreg>>, BuiltinArg<Mreg>),
    Mlabel(String),
}

impl Default for MachInst {
    fn default() -> Self {
        MachInst::Mreturn
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, EnumString)]
pub enum Typ {
    Tint,
    Tfloat,
    Tlong,
    Tsingle,
    Tany32,
    Tany64,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum XType {
    Xbool,
    Xint8signed,
    Xint8unsigned,
    Xint16signed,
    Xint16unsigned,
    Xint,
    Xintunsigned,
    Xfloat,
    Xlong,
    Xlongunsigned,
    Xsingle,
    Xptr,
    Xcharptr,
    Xcharptrptr,
    Xintptr,
    Xfloatptr,
    Xsingleptr,
    Xfuncptr,
    Xany32,
    Xany64,
    Xvoid,
    XstructPtr(StructId),
}

pub type StructId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EdgeType {
    Embed,
    Deref,
    Assign,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FieldType {
    Scalar(MemoryChunk),
    Pointer(Box<FieldType>),
    StructPointer(StructId),
    #[allow(dead_code)]
    EmbeddedStruct(StructId),
    #[allow(dead_code)]
    Array(Box<FieldType>, usize),
    #[allow(dead_code)]
    Union(Vec<FieldType>),
    #[allow(dead_code)]
    OpaqueBlob(usize),
    Unknown,
}

impl FieldType {
    pub fn size(&self, pointer_size: usize) -> usize {
        match self {
            FieldType::Scalar(chunk) => match chunk {
                MemoryChunk::MBool | MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned => 1,
                MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned => 2,
                MemoryChunk::MInt32 | MemoryChunk::MFloat32 | MemoryChunk::MAny32 => 4,
                MemoryChunk::MInt64 | MemoryChunk::MFloat64 | MemoryChunk::MAny64 => 8,
                MemoryChunk::Unknown => 4,
            },
            FieldType::Pointer(_) | FieldType::StructPointer(_) => pointer_size,
            FieldType::EmbeddedStruct(_) => pointer_size,
            FieldType::Array(elem, count) => elem.size(pointer_size) * count,
            FieldType::Union(variants) => variants
                .iter()
                .map(|v| v.size(pointer_size))
                .max()
                .unwrap_or(4),
            FieldType::OpaqueBlob(size) => *size,
            FieldType::Unknown => 4,
        }
    }

    pub fn to_type_string(&self) -> String {
        match self {
            FieldType::Scalar(chunk) => match chunk {
                MemoryChunk::MBool => "int_IBool".to_string(),
                MemoryChunk::MInt8Signed => "int_I8".to_string(),
                MemoryChunk::MInt8Unsigned => "int_I8_unsigned".to_string(),
                MemoryChunk::MInt16Signed => "int_I16".to_string(),
                MemoryChunk::MInt16Unsigned => "int_I16_unsigned".to_string(),
                MemoryChunk::MInt32 | MemoryChunk::MAny32 => "int_I32".to_string(),
                MemoryChunk::MInt64 | MemoryChunk::MAny64 => "int_I64".to_string(),
                MemoryChunk::MFloat32 => "float_F32".to_string(),
                MemoryChunk::MFloat64 => "float_F64".to_string(),
                MemoryChunk::Unknown => "int_I32".to_string(),
            },
            FieldType::Pointer(inner) => format!("ptr_{}", inner.to_type_string()),
            FieldType::StructPointer(struct_id) => format!("ptr_struct_{:x}", struct_id),
            FieldType::EmbeddedStruct(struct_id) => format!("struct_{:x}", struct_id),
            FieldType::Array(elem, _) => format!("arr_{}", elem.to_type_string()),
            FieldType::Union(variants) => variants
                .first()
                .map(|v| v.to_type_string())
                .unwrap_or_else(|| "int_I32".to_string()),
            FieldType::OpaqueBlob(size) => format!("blob_{}", size),
            FieldType::Unknown => "int_I32".to_string(),
        }
    }
}

impl Default for FieldType {
    fn default() -> Self {
        FieldType::Unknown
    }
}

pub type Z = i64;
pub type Positive = i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MemoryChunk {
    MBool,
    MInt8Signed,
    MInt8Unsigned,
    MInt16Signed,
    MInt16Unsigned,
    MInt32,
    MInt64,
    MFloat32,
    MFloat64,
    MAny32,
    MAny64,
    #[allow(dead_code)]
    Unknown,
}

/// Decoder-authenticated direction of one scalar memory operand.  This is a
/// machine effect, not a source qualifier: in particular it never implies
/// `volatile` or an aliasing class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScalarMemoryDirection {
    Read,
    Write,
}

/// The exact value-extension semantics attached to a decoded scalar load.
/// Stores and plain MOV loads use `Plain`; signedness inferred later for a C
/// declaration is deliberately kept separate from this opcode fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScalarMemoryExtension {
    Plain,
    SignExtend,
    ZeroExtend,
    /// The architectural zero-extension performed by an x86-64 write to a
    /// 32-bit general-purpose destination.  V1 admits this only for an exact
    /// `MOV r32, m32` whose surviving value is consumed at 64-bit width; it
    /// never treats byte/word partial-register writes as full values.
    ImplicitZeroExtend,
}

/// Architectural result formation after the decoded MOV-family opcode has
/// produced its encoded destination. Every x86-64 r32 write clears the upper
/// half of the parent register, including MOVSX/MOVZX. Keeping this stage
/// separate prevents `movsx eax, byte ptr [...]` from being misrepresented as
/// a direct signed 64-bit extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScalarMemoryResultChain {
    Direct,
    ZeroUpper32,
}

/// Exact integral interpretation required by one surviving machine use of an
/// authenticated extended load.  Neutral integer operations inherit the
/// opcode-defined result interpretation; operations with signed/unsigned
/// encodings override it locally.  This is provider evidence, not a global C
/// declaration inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScalarMemoryUseType {
    Signed8,
    Unsigned8,
    Signed16,
    Unsigned16,
    Signed32,
    Unsigned32,
    Signed64,
    Unsigned64,
}

impl ScalarMemoryUseType {
    pub fn width(self) -> usize {
        match self {
            Self::Signed8 | Self::Unsigned8 => 1,
            Self::Signed16 | Self::Unsigned16 => 2,
            Self::Signed32 | Self::Unsigned32 => 4,
            Self::Signed64 | Self::Unsigned64 => 8,
        }
    }

    pub fn signed(self) -> bool {
        matches!(
            self,
            Self::Signed8 | Self::Signed16 | Self::Signed32 | Self::Signed64
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScalarMemoryUseSite {
    pub node: Node,
    /// Exact surviving RTL value consumed at this leaf. It is either the load
    /// result or the output of one authenticated transport step.
    pub value: RTLReg,
    pub required_type: ScalarMemoryUseType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScalarMemoryTransportKind {
    Move,
    LaneCast(ScalarMemoryUseType),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScalarMemoryTransport {
    pub node: Node,
    pub input: RTLReg,
    pub output: RTLReg,
    pub kind: ScalarMemoryTransportKind,
}

/// Closed, function-local def/use plan for one authenticated extension load.
/// Each listed node uses the value exactly once in final RTL, is uniquely
/// owned by `function`, and is dominated by the load definition.  Clight
/// revalidates the same one-use-per-node shape before inserting private casts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScalarMemoryUsePlan {
    pub function: Address,
    pub definition_node: Node,
    pub value: RTLReg,
    pub transports: Arc<Vec<ScalarMemoryTransport>>,
    pub sites: Arc<Vec<ScalarMemoryUseSite>>,
}

impl ScalarMemoryUsePlan {
    pub fn is_closed_v1(&self, proof: &ScalarMemoryAccessProof) -> bool {
        if proof.function != self.function
            || proof.selected_node != self.definition_node
            || proof.value != self.value
            || proof.direction != ScalarMemoryDirection::Read
            || self.sites.is_empty()
            || self.sites.len() > 128
            || self.transports.len() > 128
            || self.sites.len().saturating_add(self.transports.len()) > 128
            || !self.transports.windows(2).all(|pair| pair[0] < pair[1])
            || !self.sites.windows(2).all(|pair| pair[0] < pair[1])
        {
            return false;
        }

        // The transport graph is a bounded forest rooted at the decoded load
        // value.  A value may fan out through several exact copies/casts, but
        // no output can merge definitions or feed back into an ancestor.
        let mut output_to_input = std::collections::BTreeMap::new();
        let mut transport_nodes = std::collections::BTreeSet::new();
        for transport in self.transports.iter() {
            if transport.input == transport.output
                || transport.output == self.value
                || output_to_input
                    .insert(transport.output, transport.input)
                    .is_some()
                || !transport_nodes.insert(transport.node)
            {
                return false;
            }
        }
        let mut site_occurrences = std::collections::BTreeSet::new();
        for site in self.sites.iter() {
            if transport_nodes.contains(&site.node)
                || !site_occurrences.insert((site.node, site.value))
            {
                return false;
            }
        }

        let mut reachable = std::collections::BTreeSet::from([self.value]);
        loop {
            let before = reachable.len();
            for (output, input) in &output_to_input {
                if reachable.contains(input) {
                    reachable.insert(*output);
                }
            }
            if reachable.len() == before {
                break;
            }
            if reachable.len() > 128 {
                return false;
            }
        }
        if reachable.len() != output_to_input.len().saturating_add(1)
            || self
                .sites
                .iter()
                .any(|site| !reachable.contains(&site.value))
        {
            return false;
        }

        // Every root/output is consumed by at least one exact transport or
        // terminal obligation.  Dead transport branches and hidden values are
        // not silently serialized.
        let consumed_values: std::collections::BTreeSet<_> = self
            .transports
            .iter()
            .map(|transport| transport.input)
            .chain(self.sites.iter().map(|site| site.value))
            .collect();
        if reachable
            .iter()
            .any(|value| !consumed_values.contains(value))
        {
            return false;
        }

        self.sites
            .iter()
            .all(|site| site.required_type.width() <= proof.value_width)
    }
}

/// Closed source-shape families whose machine semantics are already fixed by
/// the selected RTL program.  These identifiers are provider-internal: the
/// source-alternative wire records only a bounded, content-authenticated
/// profile derived from one proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage4SourceKind {
    AffineAddress,
    Zeroing,
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
}

/// Exact provider boundary at which one Stage-4 source shape was proved.
/// `FinalRtlDefinition` is the ordinary selected-definition path.  A
/// destructive x86 register operation may instead be erased after the RTL
/// optimizer has resolved the destination's copy chain; that second path is
/// admitted only by the optimizer-published selected-row/dead-node proof and
/// is reintroduced solely in a private source-alternative view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage4RootBoundary {
    FinalRtlDefinition,
    EliminatedMutation,
}

impl Stage4SourceKind {
    pub fn is_compound(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Sub | Self::Mul | Self::And | Self::Or | Self::Xor
        )
    }
}

/// One exact surviving use of a Stage-4 value.  Repeated occurrences at one
/// RTL node are deliberately represented as duplicate rows and rejected while
/// the plan is built; a serialized plan is therefore sorted and unique.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Stage4UseSite {
    pub node: Node,
    pub value: RTLReg,
}

/// One exact value-preserving edge in a Stage-4 final-RTL use forest.  Only a
/// selected `Omove` may publish this row; arithmetic, casts, loads, calls and
/// address formation are terminal uses rather than silently replayed aliases.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Stage4Transport {
    pub node: Node,
    pub input: RTLReg,
    pub output: RTLReg,
}

/// Exact surviving load whose canonical assignment is the placement anchor
/// for one eliminated two-address mutation.  Stage-4 never rewrites this
/// load; each later IR boundary must reproduce it byte-for-byte before the
/// private view may append a compound statement.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Stage4PlacementLoad {
    pub chunk: MemoryChunk,
    pub addressing: Addressing,
    pub args: Args,
}

/// One exact terminal observation of an eliminated mutation carrier.  V1 is
/// deliberately closed to a sole return or store and admits no move forest.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage4TerminalUse {
    Return,
    Store {
        chunk: MemoryChunk,
        addressing: Addressing,
        args: Args,
    },
}

/// Bounded final-RTL def/use envelope for a source-shape proof.  Stage-4 does
/// not replay arbitrary SSA: the authenticated definition must dominate every
/// listed direct use and no hidden, duplicate, transported, or redefined value
/// may be omitted from this list.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Stage4UsePlan {
    pub function: Address,
    pub definition_node: Node,
    /// Canonical Clight node at which an eliminated mutation may be prepended.
    /// Surviving final-RTL definitions place at their own definition node.
    pub placement_node: Node,
    pub value: RTLReg,
    pub placement_load: Option<Stage4PlacementLoad>,
    pub terminal_use: Option<Stage4TerminalUse>,
    pub transports: Arc<Vec<Stage4Transport>>,
    pub sites: Arc<Vec<Stage4UseSite>>,
}

impl Stage4UsePlan {
    pub fn is_closed_v1(&self, proof: &Stage4SourceProof) -> bool {
        if self.function != proof.function
            || self.definition_node != proof.selected_node
            || match proof.root_boundary {
                Stage4RootBoundary::FinalRtlDefinition => {
                    self.placement_node != self.definition_node
                }
                Stage4RootBoundary::EliminatedMutation => {
                    self.placement_node == self.definition_node
                }
            }
            || self.value != proof.value
            || self.sites.is_empty()
            || self.sites.len() > 64
            || self.transports.len() > 64
            || self.sites.len().saturating_add(self.transports.len()) > 64
            || !self.transports.windows(2).all(|pair| pair[0] < pair[1])
            || !self.sites.windows(2).all(|pair| pair[0] < pair[1])
        {
            return false;
        }

        match proof.root_boundary {
            Stage4RootBoundary::FinalRtlDefinition => {
                if self.placement_load.is_some() || self.terminal_use.is_some() {
                    return false;
                }
            }
            Stage4RootBoundary::EliminatedMutation => {
                if self.placement_load.is_none()
                    || self.terminal_use.is_none()
                    || !self.transports.is_empty()
                    || self.sites.len() != 1
                {
                    return false;
                }
                let placement = self.placement_load.as_ref().expect("checked placement");
                let placement_width = match placement.chunk {
                    MemoryChunk::MInt32 | MemoryChunk::MAny32 => Some(4),
                    MemoryChunk::MInt64 | MemoryChunk::MAny64 => Some(8),
                    _ => None,
                };
                let terminal_width = match self.terminal_use.as_ref().expect("checked terminal") {
                    Stage4TerminalUse::Return => Some(proof.width),
                    Stage4TerminalUse::Store { chunk, args, .. } => {
                        if args.is_empty() || args.len() > 2 || args.contains(&self.value) {
                            return false;
                        }
                        match chunk {
                            MemoryChunk::MInt32 | MemoryChunk::MAny32 => Some(4),
                            MemoryChunk::MInt64 | MemoryChunk::MAny64 => Some(8),
                            _ => None,
                        }
                    }
                };
                if placement_width != Some(proof.width)
                    || terminal_width != Some(proof.width)
                    || placement.args.is_empty()
                    || placement.args.len() > 2
                    || placement.args.contains(&self.value)
                {
                    return false;
                }
            }
        }

        let mut output_to_input = std::collections::BTreeMap::new();
        let mut transport_nodes = std::collections::BTreeSet::new();
        for transport in self.transports.iter() {
            if transport.input == transport.output
                || transport.output == self.value
                || output_to_input
                    .insert(transport.output, transport.input)
                    .is_some()
                || !transport_nodes.insert(transport.node)
            {
                return false;
            }
        }
        let mut site_occurrences = std::collections::BTreeSet::new();
        if self.sites.iter().any(|site| {
            transport_nodes.contains(&site.node)
                || !site_occurrences.insert((site.node, site.value))
        }) {
            return false;
        }

        let mut reachable = std::collections::BTreeSet::from([self.value]);
        loop {
            let before = reachable.len();
            for (output, input) in &output_to_input {
                if reachable.contains(input) {
                    reachable.insert(*output);
                }
            }
            if reachable.len() == before {
                break;
            }
            if reachable.len() > 65 {
                return false;
            }
        }
        if reachable.len() != output_to_input.len().saturating_add(1)
            || self
                .sites
                .iter()
                .any(|site| !reachable.contains(&site.value))
        {
            return false;
        }

        let consumed_values: std::collections::BTreeSet<_> = self
            .transports
            .iter()
            .map(|transport| transport.input)
            .chain(self.sites.iter().map(|site| site.value))
            .collect();
        reachable
            .iter()
            .all(|value| consumed_values.contains(value))
    }
}

/// A decoder-, LTL-, selected-RTL-, ownership-, CFG-, and COFF-authenticated
/// source-shape opportunity.  It carries no source name, score, struct, alias,
/// or volatility claim.  `operation` is the exact selected RTL operation;
/// downstream private Clight views must match it rather than synthesizing a
/// semantically broader expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Stage4SourceProof {
    pub function: Address,
    pub origin_node: Node,
    pub selected_node: Node,
    pub kind: Stage4SourceKind,
    pub root_boundary: Stage4RootBoundary,
    pub width: usize,
    pub operation: Operation,
    pub args: Arc<Vec<RTLReg>>,
    /// Exact decoded source immediate for a two-operand register-immediate RMW
    /// instruction. `operation` remains the selected RTL operation (SUB is
    /// represented there as add of the negated immediate).
    pub source_immediate: Option<i64>,
    pub value: RTLReg,
    /// Destination of the exact selected pre-optimization row. Surviving
    /// definitions equal `value`; an eliminated two-address mutation carries
    /// its distinct fresh root definition, whose exact fixed-point web is
    /// authenticated back to the surviving `value` carrier.
    pub selected_result: RTLReg,
    /// Sorted unique parameter leaves of an affine address DAG.  Non-affine
    /// forms carry an empty vector and are never allowed to infer pointer type.
    pub address_param_leaves: Arc<Vec<RTLReg>>,
}

impl Stage4SourceProof {
    pub fn is_closed_v1(&self) -> bool {
        if self.origin_node != self.selected_node
            || !matches!(self.width, 4 | 8)
            || self.args.len() > 2
            || self
                .address_param_leaves
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return false;
        }
        match self.kind {
            Stage4SourceKind::AffineAddress => {
                self.root_boundary == Stage4RootBoundary::FinalRtlDefinition
                    && self.selected_result == self.value
                    && !self.args.is_empty()
                    && !self.args.contains(&self.value)
                    && !self.address_param_leaves.is_empty()
                    && self.source_immediate.is_none()
                    && matches!(
                        self.operation,
                        Operation::Olea(Addressing::Aindexed(_))
                            | Operation::Olea(Addressing::Aindexed2(_))
                            | Operation::Olea(Addressing::Ascaled(1 | 2 | 4 | 8, _))
                            | Operation::Olea(Addressing::Aindexed2scaled(2 | 4 | 8, _))
                    )
            }
            Stage4SourceKind::Zeroing => {
                self.root_boundary == Stage4RootBoundary::FinalRtlDefinition
                    && self.selected_result == self.value
                    && self.address_param_leaves.is_empty()
                    && self.args.is_empty()
                    && self.source_immediate.is_none()
                    && matches!(
                        self.operation,
                        Operation::Ointconst(0) | Operation::Olongconst(0)
                    )
            }
            Stage4SourceKind::Add
            | Stage4SourceKind::Sub
            | Stage4SourceKind::Mul
            | Stage4SourceKind::And
            | Stage4SourceKind::Or
            | Stage4SourceKind::Xor => {
                if self.width == 4
                    && self.source_immediate.is_some_and(|value| {
                        i64::from(value as i32) != value
                    })
                {
                    return false;
                }
                let expected_operation = match (self.kind, self.width, self.source_immediate) {
                    (Stage4SourceKind::Add, 4, None) => {
                        matches!(
                            self.operation,
                            Operation::Oadd
                                | Operation::Olea(Addressing::Aindexed2(0))
                        )
                    }
                    (Stage4SourceKind::Add, 8, None) => {
                        matches!(
                            self.operation,
                            Operation::Oaddl
                                | Operation::Oleal(Addressing::Aindexed2(0))
                        )
                    }
                    (Stage4SourceKind::Sub, 4, None) => self.operation == Operation::Osub,
                    (Stage4SourceKind::Sub, 8, None) => self.operation == Operation::Osubl,
                    (Stage4SourceKind::Mul, 4, None) => self.operation == Operation::Omul,
                    (Stage4SourceKind::Mul, 8, None) => self.operation == Operation::Omull,
                    (Stage4SourceKind::And, 4, None) => self.operation == Operation::Oand,
                    (Stage4SourceKind::And, 8, None) => self.operation == Operation::Oandl,
                    (Stage4SourceKind::Or, 4, None) => self.operation == Operation::Oor,
                    (Stage4SourceKind::Or, 8, None) => self.operation == Operation::Oorl,
                    (Stage4SourceKind::Xor, 4, None) => self.operation == Operation::Oxor,
                    (Stage4SourceKind::Xor, 8, None) => self.operation == Operation::Oxorl,
                    (Stage4SourceKind::Add, 4, Some(value)) => {
                        self.operation == Operation::Oaddimm(value)
                    }
                    (Stage4SourceKind::Add, 8, Some(value)) => {
                        self.operation == Operation::Oaddlimm(value)
                    }
                    (Stage4SourceKind::Sub, 4, Some(value)) => value.checked_neg().is_some_and(
                        |negated| self.operation == Operation::Oaddimm(negated),
                    ),
                    (Stage4SourceKind::Sub, 8, Some(value)) => value.checked_neg().is_some_and(
                        |negated| self.operation == Operation::Oaddlimm(negated),
                    ),
                    (Stage4SourceKind::Mul, 4, Some(value)) => {
                        self.operation == Operation::Omulimm(value)
                    }
                    (Stage4SourceKind::Mul, 8, Some(value)) => {
                        self.operation == Operation::Omullimm(value)
                    }
                    (Stage4SourceKind::And, 4, Some(value)) => {
                        self.operation == Operation::Oandimm(value)
                    }
                    (Stage4SourceKind::And, 8, Some(value)) => {
                        self.operation == Operation::Oandlimm(value)
                    }
                    (Stage4SourceKind::Or, 4, Some(value)) => {
                        self.operation == Operation::Oorimm(value)
                    }
                    (Stage4SourceKind::Or, 8, Some(value)) => {
                        self.operation == Operation::Oorlimm(value)
                    }
                    (Stage4SourceKind::Xor, 4, Some(value)) => {
                        self.operation == Operation::Oxorimm(value)
                    }
                    (Stage4SourceKind::Xor, 8, Some(value)) => {
                        self.operation == Operation::Oxorlimm(value)
                    }
                    _ => false,
                };
                let expected_arity = if self.source_immediate.is_some() { 1 } else { 2 };
                self.address_param_leaves.is_empty()
                    && self.root_boundary == Stage4RootBoundary::EliminatedMutation
                    && self.selected_result != self.value
                    && expected_operation
                    && self.args.len() == expected_arity
                    && self.args.first() == Some(&self.value)
            }
        }
    }
}

/// Closed provider-internal spelling carried alongside an authenticated
/// scalar-memory Clight candidate.  These are selection provenance tags, not
/// source-alternative wire identifiers: the wire records only the final
/// typed-lvalue boundary that was actually assembled and emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScalarLvalueSourceForm {
    RawByte,
    TypedScaled,
}

/// Placement policy attached to a private scalar-lvalue statement candidate.
/// `Plain` preserves the Stage-2 behavior. Extension candidates are never
/// admitted to the canonical relation: the private feature view either keeps
/// a narrow transport temporary and casts every authenticated use, or hoists
/// the architectural extension into the definition-local temporary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScalarLvaluePlacement {
    Plain,
    ExtensionPerUse,
    ExtensionHoisted,
}

/// Closed proof that one surviving post-optimization RTL memory effect is the
/// reversible lowering of one real decoded MOV-family operand.  The record is
/// provider-internal and carries no source-level struct, array, volatile,
/// union, or alias claim.  V1 authorizes only raw unsigned-byte address
/// arithmetic and exact typed scaled-index source alternatives.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScalarMemoryAccessProof {
    pub function: Address,
    pub origin_node: Node,
    pub selected_node: Node,
    pub operand: Symbol,
    pub direction: ScalarMemoryDirection,
    pub extension: ScalarMemoryExtension,
    /// Width of the decoded destination register for a load. Stores have no
    /// destination. This remains 4 for an implicit x64 r32->r64 zero extend,
    /// allowing every later handoff to distinguish it from an encoded r64
    /// destination even though `value_width` is the architectural result.
    pub encoded_destination_width: Option<usize>,
    pub result_chain: Option<ScalarMemoryResultChain>,
    pub address_size: u8,
    pub base_register: Mreg,
    pub index_register: Option<Mreg>,
    pub scale: i64,
    pub displacement: i64,
    pub width: usize,
    pub value_width: usize,
    /// Exact architectural result width for a decoded load destination.
    /// Plain MOV also authenticates it against early downstream type rows;
    /// MOVSX/MOVZX are revalidated after final type/signature reconciliation,
    /// where their narrow chunk artifact is replaced by the opcode-defined
    /// result type. Stores have no destination and carry None. A sealed
    /// `ZeroUpper32` result chain is the sole case where an encoded r32 write
    /// may flow into an authenticated 64-bit C use.
    pub downstream_value_width: Option<usize>,
    pub chunk: MemoryChunk,
    pub base_value: Option<RTLReg>,
    pub index_value: Option<RTLReg>,
    pub value: RTLReg,
    /// Sorted, unique parameter leaves reached by the authenticated address
    /// DAG.  Cshminor revalidates every leaf against the reconciled final
    /// function-parameter relation before exposing a source candidate.
    pub address_param_leaves: Arc<Vec<RTLReg>>,
    pub synthetic_stack_origin: bool,
    /// A typed pointer index has exactly the decoded byte scale.  No array
    /// identity is inferred; this merely admits an address-equivalent source
    /// spelling as a bounded candidate.
    pub exact_scaled_index: bool,
}

impl ScalarMemoryAccessProof {
    /// Revalidate the sealed v1 descriptor at every IR handoff.  This does not
    /// recreate decoder evidence; it prevents a stale or cross-node relation
    /// row from changing width, extension, address shape, or synthetic-node
    /// identity after authentication.
    pub fn is_closed_v1(&self) -> bool {
        if !matches!(self.address_size, 4 | 8)
            || !matches!(self.width, 1 | 2 | 4 | 8)
            || !matches!(self.value_width, 1 | 2 | 4 | 8)
            || self.base_register.is_unknown()
            || self.base_value.is_none()
            || self.index_register.is_some() != self.index_value.is_some()
            || self.index_register.is_some_and(|index| index.is_unknown())
            || self.address_param_leaves.is_empty()
            || self
                .address_param_leaves
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return false;
        }
        if match self.direction {
            ScalarMemoryDirection::Read => {
                self.downstream_value_width != Some(self.value_width)
                    || !matches!(self.encoded_destination_width, Some(1 | 2 | 4 | 8))
                    || match self.result_chain {
                        Some(ScalarMemoryResultChain::Direct) => {
                            self.encoded_destination_width != Some(self.value_width)
                        }
                        Some(ScalarMemoryResultChain::ZeroUpper32) => {
                            self.encoded_destination_width != Some(4) || self.value_width != 8
                        }
                        None => true,
                    }
            }
            ScalarMemoryDirection::Write => {
                self.downstream_value_width.is_some()
                    || self.encoded_destination_width.is_some()
                    || self.result_chain.is_some()
            }
        } {
            return false;
        }
        match (self.index_register, self.scale) {
            (None, 1) | (Some(_), 1 | 2 | 4 | 8) => {}
            _ => return false,
        }

        let synthetic_mask = (1u64 << 62) | (1u64 << 63);
        if self.origin_node & synthetic_mask != 0
            || if self.synthetic_stack_origin {
                self.address_size != 8
                    || !matches!(self.base_register, Mreg::SP | Mreg::BP)
                    || self.index_register.is_none()
                    || self.selected_node != (self.origin_node | (1u64 << 62))
            } else {
                self.selected_node != self.origin_node
                    || (self.address_size == 8 && matches!(self.base_register, Mreg::SP | Mreg::BP))
            }
        {
            return false;
        }

        let semantic_shape = match (self.direction, self.extension, self.width) {
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::Plain, 4) => {
                self.encoded_destination_width == Some(4)
                    && self.value_width == 4
                    && self.result_chain == Some(ScalarMemoryResultChain::Direct)
                    && self.chunk == MemoryChunk::MInt32
            }
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::Plain, 8) => {
                self.encoded_destination_width == Some(8)
                    && self.value_width == 8
                    && self.result_chain == Some(ScalarMemoryResultChain::Direct)
                    && matches!(self.chunk, MemoryChunk::MInt64 | MemoryChunk::MAny64)
            }
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::Plain, 1) => {
                self.encoded_destination_width == Some(1)
                    && self.value_width == 1
                    && self.result_chain == Some(ScalarMemoryResultChain::Direct)
                    && matches!(
                        self.chunk,
                        MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned
                    )
            }
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::Plain, 2) => {
                self.encoded_destination_width == Some(2)
                    && self.value_width == 2
                    && self.result_chain == Some(ScalarMemoryResultChain::Direct)
                    && matches!(
                        self.chunk,
                        MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned
                    )
            }
            (ScalarMemoryDirection::Write, ScalarMemoryExtension::Plain, width) => {
                self.value_width == width
                    && match width {
                        1 => self.chunk == MemoryChunk::MInt8Unsigned,
                        2 => self.chunk == MemoryChunk::MInt16Unsigned,
                        4 => self.chunk == MemoryChunk::MInt32,
                        8 => matches!(self.chunk, MemoryChunk::MInt64 | MemoryChunk::MAny64),
                        _ => return false,
                    }
            }
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::SignExtend, 1) => {
                ((self.encoded_destination_width == Some(self.value_width)
                    && self.result_chain == Some(ScalarMemoryResultChain::Direct)
                    && matches!(self.value_width, 4 | 8))
                    || (self.encoded_destination_width == Some(4)
                        && self.value_width == 8
                        && self.result_chain == Some(ScalarMemoryResultChain::ZeroUpper32)))
                    && matches!(
                        self.chunk,
                        MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned
                    )
            }
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::SignExtend, 2) => {
                ((self.encoded_destination_width == Some(self.value_width)
                    && self.result_chain == Some(ScalarMemoryResultChain::Direct)
                    && self.value_width > 2)
                    || (self.encoded_destination_width == Some(4)
                        && self.value_width == 8
                        && self.result_chain == Some(ScalarMemoryResultChain::ZeroUpper32)))
                    && matches!(
                        self.chunk,
                        MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned
                    )
            }
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::SignExtend, 4) => {
                self.encoded_destination_width == Some(8)
                    && self.value_width == 8
                    && self.result_chain == Some(ScalarMemoryResultChain::Direct)
                    && self.chunk == MemoryChunk::MInt32
            }
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::ZeroExtend, 1) => {
                ((self.encoded_destination_width == Some(self.value_width)
                    && self.result_chain == Some(ScalarMemoryResultChain::Direct)
                    && matches!(self.value_width, 4 | 8))
                    || (self.encoded_destination_width == Some(4)
                        && self.value_width == 8
                        && self.result_chain == Some(ScalarMemoryResultChain::ZeroUpper32)))
                    && matches!(
                        self.chunk,
                        MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned
                    )
            }
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::ZeroExtend, 2) => {
                ((self.encoded_destination_width == Some(self.value_width)
                    && self.result_chain == Some(ScalarMemoryResultChain::Direct)
                    && self.value_width > 2)
                    || (self.encoded_destination_width == Some(4)
                        && self.value_width == 8
                        && self.result_chain == Some(ScalarMemoryResultChain::ZeroUpper32)))
                    && matches!(
                        self.chunk,
                        MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned
                    )
            }
            (
                ScalarMemoryDirection::Read,
                ScalarMemoryExtension::ImplicitZeroExtend,
                4,
            ) => {
                self.encoded_destination_width == Some(4)
                    && self.value_width == 8
                    && self.result_chain == Some(ScalarMemoryResultChain::ZeroUpper32)
                    && self.chunk == MemoryChunk::MInt32
            }
            _ => false,
        };
        if !semantic_shape {
            return false;
        }

        let exact_scaled_index = self.address_size == 8
            && self.index_value.is_some()
            && (self.displacement == 0 || self.synthetic_stack_origin)
            && self.scale == self.width as i64;
        self.exact_scaled_index == exact_scaled_index
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Signature {
    pub sig_args: Arc<Vec<XType>>,
    pub sig_res: XType,
    pub sig_cc: CallConv,
}
impl Default for Signature {
    fn default() -> Self {
        Signature {
            sig_args: Arc::new(vec![]),
            sig_res: XType::Xvoid,
            sig_cc: CallConv::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CallConv {
    pub varargs: Option<i64>,
    pub unproto: bool,
    pub structured_ret: bool,
}
impl Default for CallConv {
    fn default() -> Self {
        CallConv {
            varargs: None,
            unproto: false,
            structured_ret: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum ExternalFunction {
    EFExternal(Arc<str>, Signature),
    EFBuiltin(Arc<str>, Signature),
    EFRuntime(Arc<str>, Signature),
    EFVLoad(MemoryChunk),
    EFVStore(MemoryChunk),
    EFMalloc,
    EFFree,
    EFMemcpy(Positive, Positive),
    EFAnnot(Positive, Arc<str>, Vec<Typ>),
    EFAnnotVal(Positive, Arc<str>, Typ),
    EFInlineAsm(Arc<str>, Signature, Vec<String>),
    EFDebug(Positive, Ident, Vec<Typ>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(dead_code)]
pub enum BuiltinArg<T> {
    BA(T),
    BAInt(i64),
    BALong(i64),
    BAFloat(F64),
    BASingle(F32),
    BALoadStack(MemoryChunk, Ptrofs),
    BAAddrStack(Ptrofs),
    BALoadGlobal(MemoryChunk, Ident, Ptrofs),
    BAAddrGlobal(Ident, Ptrofs),
    BASplitLong(Box<BuiltinArg<T>>, Box<BuiltinArg<T>>),
    BAAddPtr(Box<BuiltinArg<T>>, Box<BuiltinArg<T>>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operand {
    Register(&'static str),
    Immediate(i64),
    Memory(Box<Operand>, &'static str),
    Symbol(&'static str),
}

impl From<&'static str> for Operand {
    fn from(s: &'static str) -> Self {
        if s.starts_with("$") {
            let s = s.replace("$", "");
            Operand::Immediate(s.parse::<i64>().unwrap())
        } else if s.contains("(") {
            let parts: Vec<&str> = s.split('(').collect();
            let offset = parts[0].trim();
            let base = parts[1].trim_end_matches(')');
            Operand::Memory(Box::new(Operand::from(base)), offset)
        } else if s.starts_with("%") {
            Operand::Register(s)
        } else {
            Operand::Symbol(s)
        }
    }
}

pub fn condition_for_testcond(test: TestCond) -> Condition {
    condition_for_testcond_sized(test, false)
}

pub fn condition_for_testcond_sized(test: TestCond, is_64bit: bool) -> Condition {
    match test {
        TestCond::CondE => {
            if is_64bit {
                Condition::Ccompl(Comparison::Ceq)
            } else {
                Condition::Ccomp(Comparison::Ceq)
            }
        }
        TestCond::CondNe => {
            if is_64bit {
                Condition::Ccompl(Comparison::Cne)
            } else {
                Condition::Ccomp(Comparison::Cne)
            }
        }
        TestCond::CondB => {
            if is_64bit {
                Condition::Ccomplu(Comparison::Clt)
            } else {
                Condition::Ccompu(Comparison::Clt)
            }
        }
        TestCond::CondBe => {
            if is_64bit {
                Condition::Ccomplu(Comparison::Cle)
            } else {
                Condition::Ccompu(Comparison::Cle)
            }
        }
        TestCond::CondA => {
            if is_64bit {
                Condition::Ccomplu(Comparison::Cgt)
            } else {
                Condition::Ccompu(Comparison::Cgt)
            }
        }
        TestCond::CondAe => {
            if is_64bit {
                Condition::Ccomplu(Comparison::Cge)
            } else {
                Condition::Ccompu(Comparison::Cge)
            }
        }
        TestCond::CondL => {
            if is_64bit {
                Condition::Ccompl(Comparison::Clt)
            } else {
                Condition::Ccomp(Comparison::Clt)
            }
        }
        TestCond::CondLe => {
            if is_64bit {
                Condition::Ccompl(Comparison::Cle)
            } else {
                Condition::Ccomp(Comparison::Cle)
            }
        }
        TestCond::CondG => {
            if is_64bit {
                Condition::Ccompl(Comparison::Cgt)
            } else {
                Condition::Ccomp(Comparison::Cgt)
            }
        }
        TestCond::CondGe => {
            if is_64bit {
                Condition::Ccompl(Comparison::Cge)
            } else {
                Condition::Ccomp(Comparison::Cge)
            }
        }
        TestCond::CondNp => Condition::Cmasknotzero(0),
        TestCond::CondP => Condition::Cmaskzero(0),
        // OF is a single flag bit, not a comparison: lift to the opaque overflow conditions, width-agnostic.
        TestCond::CondO => Condition::Coverflow,
        TestCond::CondNo => Condition::Cnotoverflow,
        TestCond::Unknown => Condition::Ccomp(Comparison::Unknown),
    }
}

pub fn negate_testcond(c: TestCond) -> TestCond {
    match c {
        TestCond::CondE => TestCond::CondNe,
        TestCond::CondNe => TestCond::CondE,
        TestCond::CondB => TestCond::CondAe,
        TestCond::CondBe => TestCond::CondA,
        TestCond::CondAe => TestCond::CondB,
        TestCond::CondA => TestCond::CondBe,
        TestCond::CondL => TestCond::CondGe,
        TestCond::CondLe => TestCond::CondG,
        TestCond::CondGe => TestCond::CondL,
        TestCond::CondG => TestCond::CondLe,
        TestCond::CondP => TestCond::CondNp,
        TestCond::CondNp => TestCond::CondP,
        TestCond::CondO => TestCond::CondNo,
        TestCond::CondNo => TestCond::CondO,
        TestCond::Unknown => TestCond::Unknown,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slot {
    Local,
    Incoming,
    #[allow(dead_code)]
    Outgoing,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LinearInst {
    Lgetstack(Slot, Z, Typ, Mreg),
    Lsetstack(Mreg, Slot, Z, Typ),
    Lop(Operation, MregArgs, Mreg),
    Lload(MemoryChunk, Addressing, MregArgs, Mreg),
    Lstore(MemoryChunk, Addressing, MregArgs, Mreg),
    Lcall(Either<Mreg, Either<Symbol, u64>>),
    Ltailcall(Either<Mreg, Either<Symbol, u64>>),
    Lbuiltin(String, Vec<BuiltinArg<Mreg>>, BuiltinArg<Mreg>),
    Llabel(String),
    Lgoto(Symbol),
    Lcond(Condition, MregArgs, Symbol),
    #[allow(dead_code)]
    Ljumptable(Mreg, Vec<Symbol>),
    Lreturn,
}

pub type Node = u64;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LTLInst {
    Lop(Operation, MregArgs, Mreg),
    Lload(MemoryChunk, Addressing, MregArgs, Mreg),
    Lgetstack(Slot, Z, Typ, Mreg),
    Lsetstack(Mreg, Slot, Z, Typ),
    Lstore(MemoryChunk, Addressing, MregArgs, Mreg),
    Lcall(Either<Mreg, Either<u64, Symbol>>),
    Ltailcall(Either<Mreg, Either<u64, Symbol>>),
    Lbuiltin(String, Vec<BuiltinArg<Mreg>>, BuiltinArg<Mreg>),
    Lbranch(Either<Symbol, Node>),
    Lcond(
        Condition,
        MregArgs,
        Either<Symbol, Node>,
        Either<Symbol, Node>,
    ),
    Ljumptable(Mreg, Vec<Node>),
    Lreturn,
}

pub type RTLReg = u64;

pub type Args = Arc<Vec<RTLReg>>;

pub type Targets = Arc<Vec<Node>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RTLInst {
    Inop,
    Iop(Operation, Args, RTLReg),
    Iload(MemoryChunk, Addressing, Args, RTLReg),
    Istore(MemoryChunk, Addressing, Args, RTLReg),
    Icall(
        Option<Signature>,
        Either<RTLReg, Either<u64, Symbol>>,
        Args,
        Option<RTLReg>,
        Node,
    ),
    Itailcall(Option<Signature>, Either<RTLReg, Either<u64, Symbol>>, Args),
    Ibuiltin(String, Vec<BuiltinArg<RTLReg>>, BuiltinArg<RTLReg>),
    Icond(Condition, Args, Either<Symbol, Node>, Either<Symbol, Node>),
    Ijumptable(RTLReg, Targets),
    Ibranch(Either<Symbol, Node>),
    Ireturn(RTLReg),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constant {
    Ointconst(i64),
    Ofloatconst(F64),
    Osingleconst(F32),
    Olongconst(i64),
    Oaddrsymbol(Ident, Ptrofs),
    Oaddrstack(Ptrofs),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum CminorUnop {
    Ocast8unsigned,
    Ocast8signed,
    Ocast16unsigned,
    Ocast16signed,
    Onegint,
    Onotint,
    Onegf,
    Oabsf,
    Onegfs,
    Oabsfs,
    Osingleoffloat,
    Ofloatofsingle,
    Ointoffloat,
    Ointuoffloat,
    Ofloatofint,
    Ofloatofintu,
    Ointofsingle,
    Ointuofsingle,
    Osingleofint,
    Osingleofintu,
    Onegl,
    Onotl,
    Ointoflong,
    Ointuoflong,
    Olongofint,
    Olongofintu,
    Olongoffloat,
    Olonguoffloat,
    Ofloatoflong,
    Ofloatoflongu,
    Olongofsingle,
    Olonguofsingle,
    Osingleoflong,
    Osingleoflongu,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CminorBinop {
    Oadd,
    Osub,
    Omul,
    Odiv,
    Odivu,
    Omod,
    Omodu,
    Oand,
    Oor,
    Oxor,
    Oshl,
    Oshr,
    Oshru,
    Oaddf,
    Osubf,
    Omulf,
    Odivf,
    Oaddfs,
    Osubfs,
    Omulfs,
    Odivfs,
    Omaxf,
    Ominf,
    Oaddl,
    Osubl,
    Omull,
    Odivl,
    Odivlu,
    Omodl,
    Omodlu,
    Oandl,
    Oorl,
    Oxorl,
    Oshll,
    Oshrl,
    Oshrlu,
    Omulhs,
    Omulhu,
    Omullhs,
    Omullhu,
    Ocmp(Comparison),
    Ocmpu(Comparison),
    Ocmpf(Comparison),
    Ocmpnotf(Comparison),
    Ocmpfs(Comparison),
    Ocmpnotfs(Comparison),
    Ocmpl(Comparison),
    Ocmplu(Comparison),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CminorExpr {
    Evar(RTLReg),
    Econst(Constant),
    Eunop(CminorUnop, RTLReg),
    Ebinop(CminorBinop, RTLReg, RTLReg),
    Eop(Operation, Args),
    Eload(MemoryChunk, Addressing, Args),
    #[allow(dead_code)]
    Eexternal(u64, Option<Signature>, Args),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CminorStmt {
    Sassign(RTLReg, CminorExpr),
    Sstore(MemoryChunk, Addressing, Args, RTLReg),
    Scall(
        Option<RTLReg>,
        Option<Signature>,
        Either<RTLReg, Either<u64, Symbol>>,
        Args,
    ),
    Stailcall(Option<Signature>, Either<RTLReg, Either<u64, Symbol>>, Args),
    Sbuiltin(
        Option<RTLReg>,
        String,
        Vec<BuiltinArg<RTLReg>>,
        BuiltinArg<RTLReg>,
    ),
    Sifthenelse(Condition, Args, Node, Node),
    #[allow(dead_code)]
    Sbranch(Condition, Args, Node, Node),
    Sjumptable(RTLReg, Targets),
    Sjump(Node),
    Sreturn(RTLReg),
    Snop,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CsharpminorExpr {
    Evar(RTLReg),
    #[allow(dead_code)]
    Eaddrof(Ident),
    Econst(Constant),
    Eunop(CminorUnop, Box<CsharpminorExpr>),
    Ebinop(CminorBinop, Box<CsharpminorExpr>, Box<CsharpminorExpr>),
    Eload(MemoryChunk, Box<CsharpminorExpr>),
    Econdition(
        Box<CsharpminorExpr>,
        Box<CsharpminorExpr>,
        Box<CsharpminorExpr>,
    ),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CsharpminorStmt {
    Sset(RTLReg, CsharpminorExpr),
    Sstore(MemoryChunk, CsharpminorExpr, CsharpminorExpr),
    Scall(
        Option<RTLReg>,
        Option<Signature>,
        Either<CsharpminorExpr, Either<u64, Symbol>>,
        Vec<CsharpminorExpr>,
    ),
    Stailcall(
        Option<Signature>,
        Either<CsharpminorExpr, Either<u64, Symbol>>,
        Vec<CsharpminorExpr>,
    ),
    Sbuiltin(
        Option<RTLReg>,
        String,
        Vec<BuiltinArg<CsharpminorExpr>>,
        BuiltinArg<CsharpminorExpr>,
    ),
    Scond(Condition, Vec<CsharpminorExpr>, Node, Node),
    #[allow(dead_code)]
    Sloophead(Node),
    Sjumptable(CsharpminorExpr, Targets),
    Sjump(Node),
    Sreturn(CsharpminorExpr),
    Sseq(Vec<CsharpminorStmt>),
    Snop,
    // Structured control flow (matching CompCert Csharpminor)
    Sifthenelse(
        Condition,
        Vec<CsharpminorExpr>,
        Box<CsharpminorStmt>,
        Box<CsharpminorStmt>,
    ),
    Sloop(Box<CsharpminorStmt>),
    Sbreak,
    Scontinue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClightSignedness {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClightIntSize {
    I8,
    I16,
    I32,
    IBool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClightFloatSize {
    F32,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClightAttr {
    pub attr_volatile: bool,
    pub attr_alignas: Option<u64>,
}

impl Default for ClightAttr {
    fn default() -> Self {
        ClightAttr {
            attr_volatile: false,
            attr_alignas: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClightType {
    Tvoid,
    Tint(ClightIntSize, ClightSignedness, ClightAttr),
    Tlong(ClightSignedness, ClightAttr),
    // __int128, used ONLY for the 64x64->128 high multiply; never a selected register/variable type.
    Tint128(ClightSignedness, ClightAttr),
    Tfloat(ClightFloatSize, ClightAttr),
    Tpointer(Arc<ClightType>, ClightAttr),
    #[allow(dead_code)]
    Tarray(Arc<ClightType>, Z, ClightAttr),
    Tfunction(Arc<Vec<ClightType>>, Arc<ClightType>, CallConv),
    Tstruct(Ident, ClightAttr),
    #[allow(dead_code)]
    Tunion(Ident, ClightAttr),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClightUnaryOp {
    Onotbool,
    Onotint,
    Oneg,
    Oabsfloat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClightBinaryOp {
    Oadd,
    Osub,
    Omul,
    Odiv,
    Omod,
    Oand,
    Oor,
    Oxor,
    Oshl,
    Oshr,
    Oeq,
    One,
    Olt,
    Ogt,
    Ole,
    Oge,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ClightFloat64(pub f64);

impl PartialEq for ClightFloat64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for ClightFloat64 {}

impl Hash for ClightFloat64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f64> for ClightFloat64 {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl From<ClightFloat64> for f64 {
    fn from(value: ClightFloat64) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ClightFloat32(pub f32);

impl PartialEq for ClightFloat32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for ClightFloat32 {}

impl Hash for ClightFloat32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f32> for ClightFloat32 {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<ClightFloat32> for f32 {
    fn from(value: ClightFloat32) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClightExpr {
    EconstInt(i32, ClightType),
    EconstFloat(ClightFloat64, ClightType),
    EconstSingle(ClightFloat32, ClightType),
    EconstLong(i64, ClightType),
    Evar(Ident, ClightType),
    EvarSymbol(String, ClightType),
    Etempvar(Ident, ClightType),
    Ederef(Box<ClightExpr>, ClightType),
    Eaddrof(Box<ClightExpr>, ClightType),
    Eunop(ClightUnaryOp, Box<ClightExpr>, ClightType),
    Ebinop(ClightBinaryOp, Box<ClightExpr>, Box<ClightExpr>, ClightType),
    Ecast(Box<ClightExpr>, ClightType),
    Efield(Box<ClightExpr>, Ident, ClightType),
    #[allow(dead_code)]
    Esizeof(ClightType, ClightType),
    #[allow(dead_code)]
    Ealignof(ClightType, ClightType),
    Econdition(
        Box<ClightExpr>,
        Box<ClightExpr>,
        Box<ClightExpr>,
        ClightType,
    ),
}

pub type ClightLabeledStatements = Vec<(Option<Z>, ClightStmt)>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClightStmt {
    Sskip,
    Sassign(ClightExpr, ClightExpr),
    Sset(Ident, ClightExpr),
    Scall(Option<Ident>, ClightExpr, Vec<ClightExpr>),
    #[allow(dead_code)]
    Sbuiltin(
        Option<Ident>,
        ExternalFunction,
        Vec<ClightType>,
        Vec<ClightExpr>,
    ),
    Ssequence(Vec<ClightStmt>),
    Sifthenelse(ClightExpr, Box<ClightStmt>, Box<ClightStmt>),
    Sloop(Box<ClightStmt>, Box<ClightStmt>),
    Sbreak,
    Scontinue,
    Sreturn(Option<ClightExpr>),
    Sswitch(ClightExpr, ClightLabeledStatements),
    Slabel(Ident, Box<ClightStmt>),
    Sgoto(Ident),
}

pub fn is_long_operation(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Olongconst(_)
            | Operation::Oaddl
            | Operation::Oaddlimm(_)
            | Operation::Osubl
            | Operation::Omull
            | Operation::Omullimm(_)
            | Operation::Omullhs
            | Operation::Omullhu
            | Operation::Odivl
            | Operation::Odivlu
            | Operation::Omodl
            | Operation::Omodlu
            | Operation::Odivlimm(_)
            | Operation::Odivluimm(_)
            | Operation::Omodlimm(_)
            | Operation::Omodluimm(_)
            | Operation::Oandl
            | Operation::Oandlimm(_)
            | Operation::Oorl
            | Operation::Oorlimm(_)
            | Operation::Oxorl
            | Operation::Oxorlimm(_)
            | Operation::Onotl
            | Operation::Oshll
            | Operation::Oshllimm(_)
            | Operation::Oshrl
            | Operation::Oshrlimm(_)
            | Operation::Oshrlu
            | Operation::Oshrluimm(_)
            | Operation::Oshrxlimm(_)
            | Operation::Ororlimm(_)
            | Operation::Onegl
            | Operation::Ocast32signed
            | Operation::Ocast32unsigned
            | Operation::Omakelong
            | Operation::Oleal(_)
            | Operation::Olongoffloat
            | Operation::Olongofsingle
    )
}

pub fn is_int_operation(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Ointconst(_)
            | Operation::Oadd
            | Operation::Oaddimm(_)
            | Operation::Osub
            | Operation::Omul
            | Operation::Omulimm(_)
            | Operation::Omulhs
            | Operation::Omulhu
            | Operation::Odiv
            | Operation::Odivu
            | Operation::Omod
            | Operation::Omodu
            | Operation::Odivimm(_)
            | Operation::Odivuimm(_)
            | Operation::Omodimm(_)
            | Operation::Omoduimm(_)
            | Operation::Oand
            | Operation::Oandimm(_)
            | Operation::Oor
            | Operation::Oorimm(_)
            | Operation::Oxor
            | Operation::Oxorimm(_)
            | Operation::Onot
            | Operation::Oshl
            | Operation::Oshlimm(_)
            | Operation::Oshr
            | Operation::Oshrimm(_)
            | Operation::Oshru
            | Operation::Oshruimm(_)
            | Operation::Oshrximm(_)
            | Operation::Ororimm(_)
            | Operation::Oshldimm(_)
            | Operation::Oneg
            | Operation::Ocast8signed
            | Operation::Ocast8unsigned
            | Operation::Ocast16signed
            | Operation::Ocast16unsigned
            | Operation::Olowlong
            | Operation::Ohighlong
            | Operation::Ointoffloat
            | Operation::Ointofsingle
    )
}

pub fn is_float_operation(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Ofloatconst(_)
            | Operation::Onegf
            | Operation::Oabsf
            | Operation::Oaddf
            | Operation::Osubf
            | Operation::Omulf
            | Operation::Odivf
            | Operation::Omaxf
            | Operation::Ominf
            | Operation::Ofloatofsingle
            | Operation::Ofloatofint
            | Operation::Ofloatoflong
    )
}

pub fn is_single_operation(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Osingleconst(_)
            | Operation::Onegfs
            | Operation::Oabsfs
            | Operation::Oaddfs
            | Operation::Osubfs
            | Operation::Omulfs
            | Operation::Odivfs
            | Operation::Osingleoffloat
            | Operation::Osingleofint
            | Operation::Osingleoflong
    )
}

pub fn is_boolean_operation(op: &Operation) -> bool {
    matches!(op, Operation::Ocmp(_))
}

pub fn is_pointer_operation(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Olea(_) | Operation::Oleal(_) | Operation::Oindirectsymbol(_)
    )
}

pub fn is_unsigned_operation(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Odivu
            | Operation::Omodu
            | Operation::Odivuimm(_)
            | Operation::Omoduimm(_)
            | Operation::Oshru
            | Operation::Oshruimm(_)
            | Operation::Omulhu
            | Operation::Ocast8unsigned
            | Operation::Ocast16unsigned
            | Operation::Ocast32unsigned
            | Operation::Odivlu
            | Operation::Omodlu
            | Operation::Odivluimm(_)
            | Operation::Omodluimm(_)
            | Operation::Oshrlu
            | Operation::Oshrluimm(_)
            | Operation::Omullhu
    )
}

pub fn is_signed_operation(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Odiv
            | Operation::Omod
            | Operation::Odivimm(_)
            | Operation::Omodimm(_)
            | Operation::Oshr
            | Operation::Oshrimm(_)
            | Operation::Oshrximm(_)
            | Operation::Omulhs
            | Operation::Ocast8signed
            | Operation::Ocast16signed
            | Operation::Ocast32signed
            | Operation::Odivl
            | Operation::Omodl
            | Operation::Odivlimm(_)
            | Operation::Omodlimm(_)
            | Operation::Oshrl
            | Operation::Oshrlimm(_)
            | Operation::Oshrxlimm(_)
            | Operation::Omullhs
    )
}
