public DVRecord(RecordInputStream in) {_option_flags = in.readInt();_promptTitle = readUnicodeString(in);_errorTitle = readUnicodeString(in);_promptText = readUnicodeString(in);_errorText = readUnicodeString(in);int field_size_first_formula = in.readUShort();_not_used_1 = in.readShort();_formula1 = Formula.read(field_size_first_formula, in);int field_size_sec_formula = in.readUShort();_not_used_2 = in.readShort();_formula2 = Formula.read(field_size_sec_formula, in);_regions = new CellRangeAddressList(in);}
public String toString() {return pattern();}
public InsertInstanceRequest() {super("Ots", "2016-06-20", "InsertInstance", "ots");setMethod(MethodType.POST);}
public boolean contains(Object o) {return indexOf(o) != -1;}
public final ByteBuffer encode(String s) {return encode(CharBuffer.wrap(s));}
public boolean requiresCommitBody() {return false;}
public String getKey() {return RawParseUtils.decode(enc, buffer, keyStart, keyEnd);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0, ValueEval arg1,ValueEval arg2, ValueEval arg3, ValueEval arg4) {double result;try {double d0 = NumericFunction.singleOperandEvaluate(arg0, srcRowIndex, srcColumnIndex);double d1 = NumericFunction.singleOperandEvaluate(arg1, srcRowIndex, srcColumnIndex);double d2 = NumericFunction.singleOperandEvaluate(arg2, srcRowIndex, srcColumnIndex);double d3 = NumericFunction.singleOperandEvaluate(arg3, srcRowIndex, srcColumnIndex);double d4 = NumericFunction.singleOperandEvaluate(arg4, srcRowIndex, srcColumnIndex);result = evaluate(d0, d1, d2, d3, d4 != 0.0);NumericFunction.checkValue(result);} catch (EvaluationException e) {return e.getErrorEval();}return new NumberEval(result);}
public DeleteClientVpnEndpointResult deleteClientVpnEndpoint(DeleteClientVpnEndpointRequest request) {request = beforeClientExecution(request);return executeDeleteClientVpnEndpoint(request);}
public Object get(CharSequence key) {List<TernaryTreeNode> list = autocomplete.prefixCompletion(root, key, 0);if (list == null || list.isEmpty()) {return null;}for (TernaryTreeNode n : list) {if (charSeqEquals(n.token, key)) {return n.val;}}return null;}
public StartFleetActionsResult startFleetActions(StartFleetActionsRequest request) {request = beforeClientExecution(request);return executeStartFleetActions(request);}
public CellRangeAddress getCellRangeAddress(int index) {return _list.get(index);}
public static Document loadXML(Reader is) {DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();DocumentBuilder db = null;try {db = dbf.newDocumentBuilder();}catch (Exception se) {throw new RuntimeException("Parser configuration error", se);}org.w3c.dom.Document doc = null;try {doc = db.parse(new InputSource(is));}catch (Exception se) {throw new RuntimeException("Error parsing file:" + se, se);}return doc;}
public double get(String name, double dflt) {double vals[] = (double[]) valByRound.get(name);if (vals != null) {return vals[roundNumber % vals.length];}String sval = props.getProperty(name, "" + dflt);if (sval.indexOf(":") < 0) {return Double.parseDouble(sval);}int k = sval.indexOf(":");String colName = sval.substring(0, k);sval = sval.substring(k + 1);colForValByRound.put(name, colName);vals = propToDoubleArray(sval);valByRound.put(name, vals);return vals[roundNumber % vals.length];}
public int getBackgroundImageId(){EscherSimpleProperty property = getOptRecord().lookup(EscherPropertyTypes.FILL__PATTERNTEXTURE);return property == null ? 0 : property.getPropertyValue();}
public TreeFilter getTreeFilter() {return treeFilter;}
public GetMemberResult getMember(GetMemberRequest request) {request = beforeClientExecution(request);return executeGetMember(request);}
public boolean canEncode() {return true;}
public ReplaceRouteResult replaceRoute(ReplaceRouteRequest request) {request = beforeClientExecution(request);return executeReplaceRoute(request);}
public ObjectId getResultTreeId() {return (resultTree == null) ? null : resultTree.toObjectId();}
public boolean equals(final Object o){boolean rval = this == o;if (!rval && (o != null) && (o.getClass() == this.getClass())){IntList other = ( IntList ) o;if (other._limit == _limit){rval = true;for (int j = 0; rval && (j < _limit); j++){rval = _array[ j ] == other._array[ j ];}}}return rval;}
public ListReusableDelegationSetsResult listReusableDelegationSets(ListReusableDelegationSetsRequest request) {request = beforeClientExecution(request);return executeListReusableDelegationSets(request);}
public String toString() {return "(" + a.toString() + " OR " + b.toString() + ")";}
public InitiateLayerUploadResult initiateLayerUpload(InitiateLayerUploadRequest request) {request = beforeClientExecution(request);return executeInitiateLayerUpload(request);}
public UpdateRepoRequest() {super("cr", "2016-06-07", "UpdateRepo", "cr");setUriPattern("/repos/[RepoNamespace]/[RepoName]");setMethod(MethodType.POST);}
public PhoneticFilterFactory(Map<String,String> args) {super(args);inject = getBoolean(args, INJECT, true);name = require(args, ENCODER);String v = get(args, MAX_CODE_LENGTH);if (v != null) {maxCodeLength = Integer.valueOf(v);} else {maxCodeLength = null;}if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public FetchCommand fetch() {return new FetchCommand(repo);}
public QueryPhraseMap searchPhrase( String fieldName, final List<TermInfo> phraseCandidate ){QueryPhraseMap root = getRootMap( fieldName );if( root == null ) return null;return root.searchPhrase( phraseCandidate );}
@Override public Iterator<Multiset.Entry<K>> iterator() {return new MultisetEntryIterator();}
public DBSnapshot deleteDBSnapshot(DeleteDBSnapshotRequest request) {request = beforeClientExecution(request);return executeDeleteDBSnapshot(request);}
public void setOutput() {output = true;}
public ByteBuffer compact() {throw new ReadOnlyBufferException();}
public XmlPullParser newPullParser() throws XmlPullParserException {if (parserClasses == null) throw new XmlPullParserException("Factory initialization was incomplete - has not tried "+classNamesLocation);if (parserClasses.size() == 0) throw new XmlPullParserException("No valid parser classes found in "+classNamesLocation);final StringBuilder issues = new StringBuilder();for (int i = 0; i < parserClasses.size(); i++) {final Class ppClass = (Class) parserClasses.get(i);try {final XmlPullParser pp = (XmlPullParser) ppClass.newInstance();for (Iterator iter = features.keySet().iterator(); iter.hasNext(); ) {final String key = (String) iter.next();final Boolean value = (Boolean) features.get(key);if(value != null && value.booleanValue()) {pp.setFeature(key, true);}}return pp;} catch(Exception ex) {issues.append (ppClass.getName () + ": "+ ex.toString ()+"; ");}}throw new XmlPullParserException ("could not create parser: "+issues);}
public DeleteAnalysisSchemeResult deleteAnalysisScheme(DeleteAnalysisSchemeRequest request) {request = beforeClientExecution(request);return executeDeleteAnalysisScheme(request);}
public ExcelExtractor(HSSFWorkbook wb) {super(wb);_wb = wb;_formatter = new HSSFDataFormatter();}
public IntBuffer put(int index, int c) {checkIndex(index);byteBuffer.putInt(index * SizeOf.INT, c);return this;}
public final byte getParameterClass(int index) {if (index >= paramClass.length) {return paramClass[paramClass.length - 1];}return paramClass[index];}
public ListEndpointsResult listEndpoints(ListEndpointsRequest request) {request = beforeClientExecution(request);return executeListEndpoints(request);}
public static CharsRef join(String[] words, CharsRefBuilder reuse) {int upto = 0;char[] buffer = reuse.chars();for (String word : words) {final int wordLen = word.length();final int needed = (0 == upto ? wordLen : 1 + upto + wordLen); if (needed > buffer.length) {reuse.grow(needed);buffer = reuse.chars();}if (upto > 0) {buffer[upto++] = SynonymMap.WORD_SEPARATOR;}word.getChars(0, wordLen, buffer, upto);upto += wordLen;}reuse.setLength(upto);return reuse.get();}
public StringBuffer insert(int index, float f) {return insert(index, Float.toString(f));}
public ShortBuffer put(short[] src, int srcOffset, int shortCount) {if (shortCount > remaining()) {throw new BufferOverflowException();}System.arraycopy(src, srcOffset, backingArray, offset + position, shortCount);position += shortCount;return this;}
public DisassociateResolverEndpointIpAddressResult disassociateResolverEndpointIpAddress(DisassociateResolverEndpointIpAddressRequest request) {request = beforeClientExecution(request);return executeDisassociateResolverEndpointIpAddress(request);}
public AcceptDirectConnectGatewayAssociationProposalResult acceptDirectConnectGatewayAssociationProposal(AcceptDirectConnectGatewayAssociationProposalRequest request) {request = beforeClientExecution(request);return executeAcceptDirectConnectGatewayAssociationProposal(request);}
public StopStackSetOperationResult stopStackSetOperation(StopStackSetOperationRequest request) {request = beforeClientExecution(request);return executeStopStackSetOperation(request);}
public CacheSubnetGroup createCacheSubnetGroup(CreateCacheSubnetGroupRequest request) {request = beforeClientExecution(request);return executeCreateCacheSubnetGroup(request);}
public CachedOrds(OrdinalsSegmentReader source, int maxDoc) throws IOException {offsets = new int[maxDoc + 1];int[] ords = new int[maxDoc]; long totOrds = 0;final IntsRef values = new IntsRef(32);for (int docID = 0; docID < maxDoc; docID++) {offsets[docID] = (int) totOrds;source.get(docID, values);long nextLength = totOrds + values.length;if (nextLength > ords.length) {if (nextLength > ArrayUtil.MAX_ARRAY_LENGTH) {throw new IllegalStateException("too many ordinals (>= " + nextLength + ") to cache");}ords = ArrayUtil.grow(ords, (int) nextLength);}System.arraycopy(values.ints, 0, ords, (int) totOrds, values.length);totOrds = nextLength;}offsets[maxDoc] = (int) totOrds;if ((double) totOrds / ords.length < 0.9) {this.ordinals = new int[(int) totOrds];System.arraycopy(ords, 0, this.ordinals, 0, (int) totOrds);} else {this.ordinals = ords;}}
public String getRawUserInfo() {return userInfo;}
@Override public Object[] toArray() {return ObjectArrays.toArrayImpl(this);}
public DescribeCompilationJobResult describeCompilationJob(DescribeCompilationJobRequest request) {request = beforeClientExecution(request);return executeDescribeCompilationJob(request);}
public String getQuery() {return decode(query);}
public CreateEnvironmentResult createEnvironment(CreateEnvironmentRequest request) {request = beforeClientExecution(request);return executeCreateEnvironment(request);}
public ParseTreeMatch match(ParseTree tree) {return matcher.match(tree, this);}
public boolean contains(CharSequence cs) {return map.containsKey(cs);}
public QueryRequest(String tableName) {setTableName(tableName);}
public boolean isRowGroupHiddenByParent(int row) {int endLevel;boolean endHidden;int endOfOutlineGroupIdx = findEndOfRowOutlineGroup(row);if (getRow(endOfOutlineGroupIdx + 1) == null) {endLevel = 0;endHidden = false;} else {endLevel = getRow(endOfOutlineGroupIdx + 1).getOutlineLevel();endHidden = getRow(endOfOutlineGroupIdx + 1).getZeroHeight();}int startLevel;boolean startHidden;int startOfOutlineGroupIdx = findStartOfRowOutlineGroup( row );if (startOfOutlineGroupIdx - 1 < 0 || getRow(startOfOutlineGroupIdx - 1) == null) {startLevel = 0;startHidden = false;} else {startLevel = getRow(startOfOutlineGroupIdx - 1).getOutlineLevel();startHidden = getRow(startOfOutlineGroupIdx - 1).getZeroHeight();}if (endLevel > startLevel) {return endHidden;}return startHidden;}
public boolean retryFailedLockFileCommit() {return true;}
public ValidateMatchmakingRuleSetResult validateMatchmakingRuleSet(ValidateMatchmakingRuleSetRequest request) {request = beforeClientExecution(request);return executeValidateMatchmakingRuleSet(request);}
public boolean get(String name, boolean dflt) {boolean vals[] = (boolean[]) valByRound.get(name);if (vals != null) {return vals[roundNumber % vals.length];}String sval = props.getProperty(name, "" + dflt);if (sval.indexOf(":") < 0) {return Boolean.valueOf(sval).booleanValue();}int k = sval.indexOf(":");String colName = sval.substring(0, k);sval = sval.substring(k + 1);colForValByRound.put(name, colName);vals = propToBooleanArray(sval);valByRound.put(name, vals);return vals[roundNumber % vals.length];}
public UpdateLinkAttributesResult updateLinkAttributes(UpdateLinkAttributesRequest request) {request = beforeClientExecution(request);return executeUpdateLinkAttributes(request);}
public NumericPayloadTokenFilter(TokenStream input, float payload, String typeMatch) {super(input);if (typeMatch == null) {throw new IllegalArgumentException("typeMatch must not be null");}thePayload = new BytesRef(PayloadHelper.encodeFloat(payload));this.typeMatch = typeMatch;}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append("[CALCCOUNT]\n");buffer.append("    .iterations     = ").append(Integer.toHexString(getIterations())).append("\n");buffer.append("[/CALCCOUNT]\n");return buffer.toString();}
public E push(E object) {addElement(object);return object;}
public LinkedHashMap(int initialCapacity, float loadFactor, boolean accessOrder) {super(initialCapacity, loadFactor);init();this.accessOrder = accessOrder;}
public TreeSet() {backingMap = new TreeMap<E, Object>();}
public long skip(long charCount) throws IOException {if (charCount < 0) {throw new IllegalArgumentException("charCount < 0: " + charCount);}synchronized (lock) {long skipped = 0;int toRead = charCount < 512 ? (int) charCount : 512;char[] charsSkipped = new char[toRead];while (skipped < charCount) {int read = read(charsSkipped, 0, toRead);if (read == -1) {return skipped;}skipped += read;if (read < toRead) {return skipped;}if (charCount - skipped < toRead) {toRead = (int) (charCount - skipped);}}return skipped;}}
public ValueEval getRef3DEval(Ref3DPxg rptg) {SheetRangeEvaluator sre = createExternSheetRefEvaluator(rptg.getSheetName(), rptg.getLastSheetName(), rptg.getExternalWorkbookNumber());return new LazyRefEval(rptg.getRow(), rptg.getColumn(), sre);}
public NewAnalyzerTask(PerfRunData runData) {super(runData);analyzerNames = new ArrayList<>();}
public boolean equals( Object o ) {return o instanceof EnglishStemmer;}
public void decode(long[] blocks, int blocksOffset, long[] values,int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final long block = blocks[blocksOffset++];valuesOffset = decode(block, values, valuesOffset);}}
public final void incRef() {ensureOpen();refCount.incrementAndGet();}
public ReplicationGroup testFailover(TestFailoverRequest request) {request = beforeClientExecution(request);return executeTestFailover(request);}
public RefWriter(Collection<Ref> refs) {this.refs = RefComparator.sort(refs);}
public ByteVector(int capacity) {if (capacity > 0) {blockSize = capacity;} else {blockSize = DEFAULT_BLOCK_SIZE;}array = new byte[blockSize];n = 0;}
public void endWorker() {if (workers.decrementAndGet() == 0)process.release();}
public DescribeVolumeStatusResult describeVolumeStatus(DescribeVolumeStatusRequest request) {request = beforeClientExecution(request);return executeDescribeVolumeStatus(request);}
public IntMapper(final int initialCapacity) {elements = new ArrayList<>(initialCapacity);valueKeyMap = new HashMap<>(initialCapacity);}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_borderType);out.writeShort(field_2_options);}
public synchronized void copyInto(Object[] elements) {System.arraycopy(elementData, 0, elements, 0, elementCount);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0, ValueEval arg1) {double s0;String s1;try {s0 = evaluateDoubleArg(arg0, srcRowIndex, srcColumnIndex);s1 = evaluateStringArg(arg1, srcRowIndex, srcColumnIndex);} catch (EvaluationException e) {return e.getErrorEval();}try {String formattedStr = formatter.formatRawCellContents(s0, -1, s1);return new StringEval(formattedStr);} catch (Exception e) {return ErrorEval.VALUE_INVALID;}}
public CustomViewSettingsRecordAggregate(RecordStream rs) {_begin = rs.getNext();if (_begin.getSid() != UserSViewBegin.sid) {throw new IllegalStateException("Bad begin record");}List<RecordBase> temp = new ArrayList<>();while (rs.peekNextSid() != UserSViewEnd.sid) {if (PageSettingsBlock.isComponentRecord(rs.peekNextSid())) {if (_psBlock != null) {if (rs.peekNextSid() == HeaderFooterRecord.sid) {_psBlock.addLateHeaderFooter((HeaderFooterRecord)rs.getNext());continue;}throw new IllegalStateException("Found more than one PageSettingsBlock in chart sub-stream, had sid: " + rs.peekNextSid());}_psBlock = new PageSettingsBlock(rs);temp.add(_psBlock);continue;}temp.add(rs.getNext());}_recs = temp;_end = rs.getNext(); if (_end.getSid() != UserSViewEnd.sid) {throw new IllegalStateException("Bad custom view settings end record");}}
public DeleteSignalingChannelResult deleteSignalingChannel(DeleteSignalingChannelRequest request) {request = beforeClientExecution(request);return executeDeleteSignalingChannel(request);}
@Override public boolean remove(Object o) {if (contains(o)) {Entry<?> entry = (Entry<?>) o;AtomicInteger frequency = backingMap.remove(entry.getElement());int numberRemoved = frequency.getAndSet(0);size -= numberRemoved;return true;}return false;}
public SnapshotDeletionPolicy(IndexDeletionPolicy primary) {this.primary = primary;}
public void throwException() throws BufferUnderflowException,BufferOverflowException, UnmappableCharacterException,MalformedInputException, CharacterCodingException {switch (this.type) {case TYPE_UNDERFLOW:throw new BufferUnderflowException();case TYPE_OVERFLOW:throw new BufferOverflowException();case TYPE_UNMAPPABLE_CHAR:throw new UnmappableCharacterException(this.length);case TYPE_MALFORMED_INPUT:throw new MalformedInputException(this.length);default:throw new CharacterCodingException();}}
public StringPtg(LittleEndianInput in)  {int nChars = in.readUByte(); _is16bitUnicode = (in.readByte() & 0x01) != 0;if (_is16bitUnicode) {field_3_string = StringUtil.readUnicodeLE(in, nChars);} else {field_3_string = StringUtil.readCompressedUnicode(in, nChars);}}
public GetPublicAccessUrlsRequest() {super("CloudPhoto", "2017-07-11", "GetPublicAccessUrls", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public CleanCommand clean() {return new CleanCommand(repo);}
public Collection<PackFile> getPacks() {PackList list = packList.get();if (list == NO_PACKS)list = scanPacks(list);PackFile[] packs = list.packs;return Collections.unmodifiableCollection(Arrays.asList(packs));}
public DescribeStackDriftDetectionStatusResult describeStackDriftDetectionStatus(DescribeStackDriftDetectionStatusRequest request) {request = beforeClientExecution(request);return executeDescribeStackDriftDetectionStatus(request);}
public ListCloudFrontOriginAccessIdentitiesResult listCloudFrontOriginAccessIdentities(ListCloudFrontOriginAccessIdentitiesRequest request) {request = beforeClientExecution(request);return executeListCloudFrontOriginAccessIdentities(request);}
public static SshSessionFactory getInstance() {return INSTANCE;}
public ListConferenceProvidersResult listConferenceProviders(ListConferenceProvidersRequest request) {request = beforeClientExecution(request);return executeListConferenceProviders(request);}
public UpdateReceiptRuleResult updateReceiptRule(UpdateReceiptRuleRequest request) {request = beforeClientExecution(request);return executeUpdateReceiptRule(request);}
public String toString() {final StringBuilder r = new StringBuilder();r.append("("); for (int i = 0; i < subfilters.length; i++) {if (i > 0)r.append(" OR "); r.append(subfilters[i].toString());}r.append(")"); return r.toString();}
public void serialize(LittleEndianOutput out) {out.writeShort(sid);out.writeShort(length);out.writeShort(flags);}
public UpdateHealthCheckResult updateHealthCheck(UpdateHealthCheckRequest request) {request = beforeClientExecution(request);return executeUpdateHealthCheck(request);}
public synchronized long ramBytesUsed() {long bytes = 0;for(CachedOrds ords : ordsCache.values()) {bytes += ords.ramBytesUsed();}return bytes;}
public UpdateWorkforceResult updateWorkforce(UpdateWorkforceRequest request) {request = beforeClientExecution(request);return executeUpdateWorkforce(request);}
public void setObjectId(AnyObjectId id) {id.copyRawTo(idBuffer(), idOffset());}
public void write(byte[] buffer, int byteOffset, int byteCount) throws IOException {IoBridge.write(fd, buffer, byteOffset, byteCount);if (syncMetadata) {fd.sync();}}
public GetBlockResult getBlock(GetBlockRequest request) {request = beforeClientExecution(request);return executeGetBlock(request);}
public void exportDirectory(File dir) {exportBase.add(dir);}
public CreateReservedInstancesListingResult createReservedInstancesListing(CreateReservedInstancesListingRequest request) {request = beforeClientExecution(request);return executeCreateReservedInstancesListing(request);}
public ByteBuffer put(byte b) {throw new ReadOnlyBufferException();}
public ValueEval evaluate(ValueEval[] args, int srcCellRow, int srcCellCol) {double result;try {List<Double> temp = new ArrayList<>();for (ValueEval arg : args) {collectValues(arg, temp);}double[] values = new double[temp.size()];for (int i = 0; i < values.length; i++) {values[i] = temp.get(i).doubleValue();}result = evaluate(values);} catch (EvaluationException e) {return e.getErrorEval();}return new NumberEval(result);}
public static int getCharType(char ch) {if (isSurrogate(ch))return CharType.SURROGATE;if (ch >= 0x4E00 && ch <= 0x9FA5)return CharType.HANZI;if ((ch >= 0x0041 && ch <= 0x005A) || (ch >= 0x0061 && ch <= 0x007A))return CharType.LETTER;if (ch >= 0x0030 && ch <= 0x0039)return CharType.DIGIT;if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == '　')return CharType.SPACE_LIKE;if ((ch >= 0x0021 && ch <= 0x00BB) || (ch >= 0x2010 && ch <= 0x2642)|| (ch >= 0x3001 && ch <= 0x301E))return CharType.DELIMITER;if ((ch >= 0xFF21 && ch <= 0xFF3A) || (ch >= 0xFF41 && ch <= 0xFF5A))return CharType.FULLWIDTH_LETTER;if (ch >= 0xFF10 && ch <= 0xFF19)return CharType.FULLWIDTH_DIGIT;if (ch >= 0xFE30 && ch <= 0xFF63)return CharType.DELIMITER;return CharType.OTHER;}
public StopJumpserverRequest() {super("HPC", "2016-06-03", "StopJumpserver", "hpc");setMethod(MethodType.POST);}
public CreateDirectoryConfigResult createDirectoryConfig(CreateDirectoryConfigRequest request) {request = beforeClientExecution(request);return executeCreateDirectoryConfig(request);}
public DescribeExportTasksResult describeExportTasks() {return describeExportTasks(new DescribeExportTasksRequest());}
public ExportClientVpnClientCertificateRevocationListResult exportClientVpnClientCertificateRevocationList(ExportClientVpnClientCertificateRevocationListRequest request) {request = beforeClientExecution(request);return executeExportClientVpnClientCertificateRevocationList(request);}
public CompleteMultipartUploadResult completeMultipartUpload(CompleteMultipartUploadRequest request) {request = beforeClientExecution(request);return executeCompleteMultipartUpload(request);}
public long ramBytesUsed() {long sizeInBytes = 0;sizeInBytes += RamUsageEstimator.sizeOf(minValues);sizeInBytes += RamUsageEstimator.sizeOf(averages);for(PackedInts.Reader reader: subReaders) {sizeInBytes += reader.ramBytesUsed();}return sizeInBytes;}
public static void fill(Object[] array, Object value) {for (int i = 0; i < array.length; i++) {array[i] = value;}}
public ByteBuffer putDouble(int index, double value) {throw new ReadOnlyBufferException();}
public DescribeAdjustmentTypesResult describeAdjustmentTypes() {return describeAdjustmentTypes(new DescribeAdjustmentTypesRequest());}
public PersonIdent getSourceCommitter() {RevCommit c = getSourceCommit();return c != null ? c.getCommitterIdent() : null;}
public Object[] toArray() {int index = 0;Object[] contents = new Object[size];Link<E> link = voidLink.next;while (link != voidLink) {contents[index++] = link.data;link = link.next;}return contents;}
public String toString() {return name + " version " + version;}
public PushCommand setRefSpecs(RefSpec... specs) {checkCallable();this.refSpecs.clear();Collections.addAll(refSpecs, specs);return this;}
public String toString(String field) {StringBuilder buffer = new StringBuilder();buffer.append("spanFirst(");buffer.append(match.toString(field));buffer.append(", ");buffer.append(end);buffer.append(")");return buffer.toString();}
public X509Certificate[] getAcceptedIssuers() {return null;}
public int read() {if (pos < size) {return s.charAt(pos++);} else {s = null;return -1;}}
public PersonIdent getRefLogIdent() {return destination.getRefLogIdent();}
@Override public int size() {return size;}
public GetRequestValidatorsResult getRequestValidators(GetRequestValidatorsRequest request) {request = beforeClientExecution(request);return executeGetRequestValidators(request);}
public String toString() {return "I(F)";}
public boolean equals(Object obj) {if (this == obj)return true;if (obj == null)return false;if (getClass() != obj.getClass())return false;SegToken other = (SegToken) obj;if (!Arrays.equals(charArray, other.charArray))return false;if (endOffset != other.endOffset)return false;if (index != other.index)return false;if (startOffset != other.startOffset)return false;if (weight != other.weight)return false;if (wordType != other.wordType)return false;return true;}
public int fillFields(byte[] data, int offset, EscherRecordFactory recordFactory) { readHeader( data, offset );int pos            = offset + 8;int size           = 0;field_1_shapeId    =  LittleEndian.getInt( data, pos + size );     size += 4;field_2_flags      =  LittleEndian.getInt( data, pos + size );     size += 4;return getRecordSize();}
public String getSignerName() {return ALGORITHM_NAME;}
public synchronized void clear() {if (size != 0) {Arrays.fill(table, null);modCount++;size = 0;}}
public CancelCapacityReservationResult cancelCapacityReservation(CancelCapacityReservationRequest request) {request = beforeClientExecution(request);return executeCancelCapacityReservation(request);}
public ImportDocumentationPartsResult importDocumentationParts(ImportDocumentationPartsRequest request) {request = beforeClientExecution(request);return executeImportDocumentationParts(request);}
public SuggestResult suggest(SuggestRequest request) {request = beforeClientExecution(request);return executeSuggest(request);}
public Explanation explain(int docId, String field, int numPayloadsSeen, float payloadScore){return Explanation.match(docScore(docId, field, numPayloadsSeen, payloadScore),getClass().getSimpleName() + ".docScore()");}
public int serialize(int offset, byte[] data) {int result = 0;for (org.apache.poi.hssf.record.Record rec : _list) {result += rec.serialize(offset + result, data);}return result;}
public String toString() {return _string.toString();}
public static long[] copyOfRange(long[] original, int start, int end) {if (start > end) {throw new IllegalArgumentException();}int originalLength = original.length;if (start < 0 || start > originalLength) {throw new ArrayIndexOutOfBoundsException();}int resultLength = end - start;int copyLength = Math.min(resultLength, originalLength - start);long[] result = new long[resultLength];System.arraycopy(original, start, result, 0, copyLength);return result;}
public static byte[] toByteArray(ByteBuffer buffer, int length) {if(buffer.hasArray() && buffer.arrayOffset() == 0) {return buffer.array();}checkByteSizeLimit(length);byte[] data = new byte[length];buffer.get(data);return data;}
public synchronized void setProgress(int progress) {setProgress(progress, false);}
public void removeCell(CellValueRecordInterface cell) {if (cell == null) {throw new IllegalArgumentException("cell must not be null");}int row = cell.getRow();if (row >= records.length) {throw new RuntimeException("cell row is out of range");}CellValueRecordInterface[] rowCells = records[row];if (rowCells == null) {throw new RuntimeException("cell row is already empty");}short column = cell.getColumn();if (column >= rowCells.length) {throw new RuntimeException("cell column is out of range");}rowCells[column] = null;}
public static String canonicalizePath(String path, boolean discardRelativePrefix) {int segmentStart = 0;int deletableSegments = 0;for (int i = 0; i <= path.length(); ) {int nextSegmentStart;if (i == path.length()) {nextSegmentStart = i;} else if (path.charAt(i) == '/') {nextSegmentStart = i + 1;} else {i++;continue;}if (i == segmentStart + 1 && path.regionMatches(segmentStart, ".", 0, 1)) {path = path.substring(0, segmentStart) + path.substring(nextSegmentStart);i = segmentStart;} else if (i == segmentStart + 2 && path.regionMatches(segmentStart, "..", 0, 2)) {if (deletableSegments > 0 || discardRelativePrefix) {deletableSegments--;int prevSegmentStart = path.lastIndexOf('/', segmentStart - 2) + 1;path = path.substring(0, prevSegmentStart) + path.substring(nextSegmentStart);i = segmentStart = prevSegmentStart;} else {i++;segmentStart = i;}} else {if (i > 0) {deletableSegments++;}i++;segmentStart = i;}}return path;}
public ApostropheFilterFactory(Map<String, String> args) {super(args);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameter(s): " + args);}}
public Entry<String, Ref> peek() {if (packedIdx < packed.size() && looseIdx < loose.size()) {Ref p = packed.get(packedIdx);Ref l = loose.get(looseIdx);int cmp = RefComparator.compareTo(p, l);if (cmp < 0) {packedIdx++;return toEntry(p);}if (cmp == 0)packedIdx++;looseIdx++;return toEntry(resolveLoose(l));}if (looseIdx < loose.size())return toEntry(resolveLoose(loose.get(looseIdx++)));if (packedIdx < packed.size())return toEntry(packed.get(packedIdx++));return null;}
public DeleteEnvironmentResult deleteEnvironment(DeleteEnvironmentRequest request) {request = beforeClientExecution(request);return executeDeleteEnvironment(request);}
public int stem(char s[], int len) {for (int i = 0; i < len; i++)switch(s[i]) {case 'á': s[i] = 'a'; break;case 'ë':case 'é': s[i] = 'e'; break;case 'í': s[i] = 'i'; break;case 'ó':case 'ő':case 'õ':case 'ö': s[i] = 'o'; break;case 'ú':case 'ű':case 'ũ':case 'û':case 'ü': s[i] = 'u'; break;}len = removeCase(s, len);len = removePossessive(s, len);len = removePlural(s, len);return normalize(s, len);}
public void addChildBefore(EscherRecord record, int insertBeforeRecordId) {int idx = 0;for (EscherRecord rec : this) {if(rec.getRecordId() == (short)insertBeforeRecordId) {break;}idx++;}_childRecords.add(idx, record);}
public ListAlbumsRequest() {super("CloudPhoto", "2017-07-11", "ListAlbums", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public SaveTaskForUpdatingRegistrantInfoByIdentityCredentialRequest() {super("Domain-intl", "2017-12-18", "SaveTaskForUpdatingRegistrantInfoByIdentityCredential", "domain");setMethod(MethodType.POST);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0) {int result;if (arg0 instanceof TwoDEval) {result = ((TwoDEval) arg0).getHeight();} else if (arg0 instanceof RefEval) {result = 1;} else { return ErrorEval.VALUE_INVALID;}return new NumberEval(result);}
public DescribeReservedInstancesResult describeReservedInstances() {return describeReservedInstances(new DescribeReservedInstancesRequest());}
public void setPackedGitMMAP(boolean usemmap) {packedGitMMAP = usemmap;}
public POIFSDocumentPath(){this.components = new String[ 0 ];}
public String toString() {return key + "/" + value;}
public void decode(byte[] blocks, int blocksOffset, int[] values, int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final int byte0 = blocks[blocksOffset++] & 0xFF;final int byte1 = blocks[blocksOffset++] & 0xFF;final int byte2 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = (byte0 << 12) | (byte1 << 4) | (byte2 >>> 4);final int byte3 = blocks[blocksOffset++] & 0xFF;final int byte4 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte2 & 15) << 16) | (byte3 << 8) | byte4;}}
public void serialize(LittleEndianOutput out) {out.writeShort(_extBookIndex);out.writeShort(_firstSheetIndex);out.writeShort(_lastSheetIndex);}
public PatternParser(PatternConsumer consumer) {this();this.consumer = consumer;}
public final String[] getValues(String name) {List<String> result = new ArrayList<>();for (IndexableField field : fields) {if (field.name().equals(name) && field.stringValue() != null) {result.add(field.stringValue());}}if (result.size() == 0) {return NO_STRINGS;}return result.toArray(new String[result.size()]);}
public ListIdentityPoolUsageResult listIdentityPoolUsage(ListIdentityPoolUsageRequest request) {request = beforeClientExecution(request);return executeListIdentityPoolUsage(request);}
public ValueEval evaluate(ValueEval[] args, int srcCellRow, int srcCellCol) {if(args.length < 1 || args.length > 5) {return ErrorEval.VALUE_INVALID;}try {BaseRef baseRef = evaluateBaseRef(args[0]);int rowOffset = (args[1] instanceof MissingArgEval) ? 0 : evaluateIntArg(args[1], srcCellRow, srcCellCol);int columnOffset = (args[2] instanceof MissingArgEval) ? 0 : evaluateIntArg(args[2], srcCellRow, srcCellCol);int height = baseRef.getHeight();int width = baseRef.getWidth();switch(args.length) {case 5:if(!(args[4] instanceof MissingArgEval)) {width = evaluateIntArg(args[4], srcCellRow, srcCellCol);}case 4:if(!(args[3] instanceof MissingArgEval)) {height = evaluateIntArg(args[3], srcCellRow, srcCellCol);}break;default:break;}if(height == 0 || width == 0) {return ErrorEval.REF_INVALID;}LinearOffsetRange rowOffsetRange = new LinearOffsetRange(rowOffset, height);LinearOffsetRange colOffsetRange = new LinearOffsetRange(columnOffset, width);return createOffset(baseRef, rowOffsetRange, colOffsetRange);} catch (EvaluationException e) {return e.getErrorEval();}}
public int[] getCountsByTime() {return countsByTime;}
public UpdateAccountResult updateAccount(UpdateAccountRequest request) {request = beforeClientExecution(request);return executeUpdateAccount(request);}
public DescribeTrainingJobResult describeTrainingJob(DescribeTrainingJobRequest request) {request = beforeClientExecution(request);return executeDescribeTrainingJob(request);}
public DeleteGroupResult deleteGroup(DeleteGroupRequest request) {request = beforeClientExecution(request);return executeDeleteGroup(request);}
public int advance(int target) {upto++;if (upto == docIDs.length) {return docID = NO_MORE_DOCS;}int inc = 10;int nextUpto = upto+10;int low;int high;while (true) {if (nextUpto >= docIDs.length) {low = nextUpto-inc;high = docIDs.length-1;break;}if (target <= docIDs[nextUpto]) {low = nextUpto-inc;high = nextUpto;break;}inc *= 2;nextUpto += inc;}while (true) {if (low > high) {upto = low;break;}int mid = (low + high) >>> 1;int cmp = docIDs[mid] - target;if (cmp < 0) {low = mid + 1;} else if (cmp > 0) {high = mid - 1;} else {upto = mid;break;}}if (upto == docIDs.length) {return docID = NO_MORE_DOCS;} else {return docID = docIDs[upto];}}
public void registerListener(final POIFSReaderListener listener) {if (listener == null) {throw new NullPointerException();}if (registryClosed) {throw new IllegalStateException();}registry.registerListener(listener);}
public static int[] grow(int[] array, int minSize) {assert minSize >= 0: "size must be positive (got " + minSize + "): likely integer overflow?";if (array.length < minSize) {return growExact(array, oversize(minSize, Integer.BYTES));} elsereturn array;}
public void visitTerminal(TerminalNode node) {System.out.println("consume "+node.getSymbol()+" rule "+getRuleNames()[_ctx.getRuleIndex()]);}
public TokenStream create(TokenStream input) {return new LatvianStemFilter(input);}
public ReplicationGroup increaseReplicaCount(IncreaseReplicaCountRequest request) {request = beforeClientExecution(request);return executeIncreaseReplicaCount(request);}
public void decode(byte[] blocks, int blocksOffset, long[] values, int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final long byte0 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = byte0 >>> 5;values[valuesOffset++] = (byte0 >>> 2) & 7;final long byte1 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte0 & 3) << 1) | (byte1 >>> 7);values[valuesOffset++] = (byte1 >>> 4) & 7;values[valuesOffset++] = (byte1 >>> 1) & 7;final long byte2 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte1 & 1) << 2) | (byte2 >>> 6);values[valuesOffset++] = (byte2 >>> 3) & 7;values[valuesOffset++] = byte2 & 7;}}
public StopHyperParameterTuningJobResult stopHyperParameterTuningJob(StopHyperParameterTuningJobRequest request) {request = beforeClientExecution(request);return executeStopHyperParameterTuningJob(request);}
public ResetNetworkInterfaceAttributeResult resetNetworkInterfaceAttribute(ResetNetworkInterfaceAttributeRequest request) {request = beforeClientExecution(request);return executeResetNetworkInterfaceAttribute(request);}
public RevBlob lookupBlob(AnyObjectId id) {RevBlob c = (RevBlob) objects.get(id);if (c == null) {c = new RevBlob(id);objects.add(c);}return c;}
public ListGroupMembershipsResult listGroupMemberships(ListGroupMembershipsRequest request) {request = beforeClientExecution(request);return executeListGroupMemberships(request);}
public static void mkdir(File d, boolean skipExisting)throws IOException {if (!d.mkdir()) {if (skipExisting && d.isDirectory())return;throw new IOException(MessageFormat.format(JGitText.get().mkDirFailed, d.getAbsolutePath()));}}
public UpdateDetectorVersionMetadataResult updateDetectorVersionMetadata(UpdateDetectorVersionMetadataRequest request) {request = beforeClientExecution(request);return executeUpdateDetectorVersionMetadata(request);}
public void write(String str, int offset, int count) throws IOException {if ((offset | count) < 0 || offset > str.length() - count) {throw new StringIndexOutOfBoundsException(str, offset, count);}char[] buf = new char[count];str.getChars(offset, offset + count, buf, 0);synchronized (lock) {write(buf, 0, buf.length);}}
public synchronized void ensureCapacity(int min) {super.ensureCapacity(min);}
public DescribeRecipeResult describeRecipe(DescribeRecipeRequest request) {request = beforeClientExecution(request);return executeDescribeRecipe(request);}
public DisassociateRouteTableResult disassociateRouteTable(DisassociateRouteTableRequest request) {request = beforeClientExecution(request);return executeDisassociateRouteTable(request);}
public SetTopicAttributesRequest(String topicArn, String attributeName, String attributeValue) {setTopicArn(topicArn);setAttributeName(attributeName);setAttributeValue(attributeValue);}
public static char[] grow(char[] array, int minSize) {assert minSize >= 0: "size must be positive (got " + minSize + "): likely integer overflow?";if (array.length < minSize) {return growExact(array, oversize(minSize, Character.BYTES));} elsereturn array;}
public StashCreateCommand setRef(String ref) {this.ref = ref;return this;}
public FormulaRecord(RecordInputStream ris) {super(ris);long valueLongBits  = ris.readLong();field_5_options = ris.readShort();specialCachedValue = FormulaSpecialCachedValue.create(valueLongBits);if (specialCachedValue == null) {field_4_value = Double.longBitsToDouble(valueLongBits);}field_6_zero = ris.readInt();int field_7_expression_len = ris.readShort(); int nBytesAvailable = ris.available();field_8_parsed_expr = Formula.read(field_7_expression_len, ris, nBytesAvailable);}
public SynonymQuery build() {Collections.sort(terms, Comparator.comparing(a -> a.term));return new SynonymQuery(terms.toArray(new TermAndBoost[0]), field);}
public PasswordRev4Record(RecordInputStream in) {field_1_password = in.readShort();}
public boolean isReadOnly() {return false;}
public int preceding(int pos) {if (pos < start || pos > end) {throw new IllegalArgumentException("offset out of bounds");} else if (pos == start) {current = start;return DONE;} else {return first();}}
public CodepageRecord(RecordInputStream in) {field_1_codepage = in.readShort();}
public ApproveAssignmentResult approveAssignment(ApproveAssignmentRequest request) {request = beforeClientExecution(request);return executeApproveAssignment(request);}
public DescribeVpnConnectionsResult describeVpnConnections() {return describeVpnConnections(new DescribeVpnConnectionsRequest());}
public final V next() { return nextEntry().value; }
public DescribeInstanceHealthResult describeInstanceHealth(DescribeInstanceHealthRequest request) {request = beforeClientExecution(request);return executeDescribeInstanceHealth(request);}
public static void register(TransportProtocol proto) {protocols.add(0, new WeakReference<>(proto));}
public static char[] copyOfRange(char[] original, int start, int end) {if (start > end) {throw new IllegalArgumentException();}int originalLength = original.length;if (start < 0 || start > originalLength) {throw new ArrayIndexOutOfBoundsException();}int resultLength = end - start;int copyLength = Math.min(resultLength, originalLength - start);char[] result = new char[resultLength];System.arraycopy(original, start, result, 0, copyLength);return result;}
public static void fill(int[] array, int value) {for (int i = 0; i < array.length; i++) {array[i] = value;}}
public Class<? extends Record> peekNextClass() {if(!hasNext()) {return null;}return _list.get(_nextIndex).getClass();}
public static char[] copyOf(char[] original, int newLength) {if (newLength < 0) {throw new NegativeArraySizeException();}return copyOfRange(original, 0, newLength);}
public DeleteRelationalDatabaseResult deleteRelationalDatabase(DeleteRelationalDatabaseRequest request) {request = beforeClientExecution(request);return executeDeleteRelationalDatabase(request);}
public boolean equals(Object obj) {if (this == obj) {return true;}if (obj == null) {return false;}if (getClass() != obj.getClass()) {return false;}WeightedPhraseInfo other = (WeightedPhraseInfo) obj;if (getStartOffset() != other.getStartOffset()) {return false;}if (getEndOffset() != other.getEndOffset()) {return false;}if (getBoost() != other.getBoost()) {return false;}return true;}
public boolean hasNext() {return nextBlock != POIFSConstants.END_OF_CHAIN;}
public void write(char b) {if (len >= buf.length) {resize(len +1);}unsafeWrite(b);}
public void serialize(LittleEndianOutput out) {futureHeader.serialize(out);out.writeShort(isf_sharedFeatureType);out.writeByte(reserved);out.writeInt((int)cbHdrData);out.write(rgbHdrData);}
public ListUserHierarchyGroupsResult listUserHierarchyGroups(ListUserHierarchyGroupsRequest request) {request = beforeClientExecution(request);return executeListUserHierarchyGroups(request);}
public GetTopicAttributesRequest(String topicArn) {setTopicArn(topicArn);}
public CreateTrafficPolicyVersionResult createTrafficPolicyVersion(CreateTrafficPolicyVersionRequest request) {request = beforeClientExecution(request);return executeCreateTrafficPolicyVersion(request);}
@Override public boolean equals(Object object) {if (this == object) {return true;}if (object instanceof Map.Entry) {Map.Entry<?, ?> entry = (Map.Entry<?, ?>) object;return (key == null ? entry.getKey() == null : key.equals(entry.getKey()))&& (value == null ? entry.getValue() == null : value.equals(entry.getValue()));}return false;}
public ListResourcesResult listResources(ListResourcesRequest request) {request = beforeClientExecution(request);return executeListResources(request);}
public final V getAndSet(V newValue) {while (true) {V x = get();if (compareAndSet(x, newValue))return x;}}
public FeatHdrRecord() {futureHeader = new FtrHeader();futureHeader.setRecordType(sid);}
public DisassociatePhoneNumbersFromVoiceConnectorResult disassociatePhoneNumbersFromVoiceConnector(DisassociatePhoneNumbersFromVoiceConnectorRequest request) {request = beforeClientExecution(request);return executeDisassociatePhoneNumbersFromVoiceConnector(request);}
public ObjectId idFor(int type, byte[] data) {return idFor(type, data, 0, data.length);}
public void removeParseListener(ParseTreeListener listener) {if (_parseListeners != null) {if (_parseListeners.remove(listener)) {if (_parseListeners.isEmpty()) {_parseListeners = null;}}}}
public AxisRecord(RecordInputStream in) {field_1_axisType  = in.readShort();field_2_reserved1 = in.readInt();field_3_reserved2 = in.readInt();field_4_reserved3 = in.readInt();field_5_reserved4 = in.readInt();}
public static double evaluate(double[] v) throws EvaluationException {if (v.length < 2) {throw new EvaluationException(ErrorEval.NA);}int[] counts = new int[v.length];Arrays.fill(counts, 1);for (int i = 0, iSize = v.length; i < iSize; i++) {for (int j = i + 1, jSize = v.length; j < jSize; j++) {if (v[i] == v[j])counts[i]++;}}double maxv = 0;int maxc = 0;for (int i = 0, iSize = counts.length; i < iSize; i++) {if (counts[i] > maxc) {maxv = v[i];maxc = counts[i];}}if (maxc > 1) {return maxv;}throw new EvaluationException(ErrorEval.NA);}
public void addFacetCount(BytesRef facetValue, int count) {if (count < currentMin) {return;}FacetEntry facetEntry = new FacetEntry(facetValue, count);if (facetEntries.size() == maxSize) {if (facetEntries.higher(facetEntry) == null) {return;}facetEntries.pollLast();}facetEntries.add(facetEntry);if (facetEntries.size() == maxSize) {currentMin = facetEntries.last().count;}}
public String toString(){StringBuilder buffer = new StringBuilder();String nl = System.getProperty("line.separator");buffer.append("[ftGmo]" + nl);buffer.append("  reserved = ").append(HexDump.toHex(reserved)).append(nl);buffer.append("[/ftGmo]" + nl);return buffer.toString();}
public String toString() {return getMode().toString() + " " + getName(); }
public CharVector(int capacity) {if (capacity > 0) {blockSize = capacity;} else {blockSize = DEFAULT_BLOCK_SIZE;}array = new char[blockSize];n = 0;}
public DescribeAccountLimitsResult describeAccountLimits(DescribeAccountLimitsRequest request) {request = beforeClientExecution(request);return executeDescribeAccountLimits(request);}
public void removeBuiltinRecord(byte name, int sheetIndex) {linkTable.removeBuiltinRecord(name, sheetIndex);}
public CreateSecurityGroupResult createSecurityGroup(CreateSecurityGroupRequest request) {request = beforeClientExecution(request);return executeCreateSecurityGroup(request);}
public boolean equals(Object other) {return sameClassAs(other) &&equalsTo(getClass().cast(other));}
public GetObjectInformationResult getObjectInformation(GetObjectInformationRequest request) {request = beforeClientExecution(request);return executeGetObjectInformation(request);}
public StringBuffer append(long l) {IntegralToString.appendLong(this, l);return this;}
public GetIntegrationResponsesResult getIntegrationResponses(GetIntegrationResponsesRequest request) {request = beforeClientExecution(request);return executeGetIntegrationResponses(request);}
public ListDeploymentConfigsResult listDeploymentConfigs() {return listDeploymentConfigs(new ListDeploymentConfigsRequest());}
public CellRangeAddress remove(int rangeIndex) {if (_list.isEmpty()) {throw new RuntimeException("List is empty");}if (rangeIndex < 0 || rangeIndex >= _list.size()) {throw new RuntimeException("Range index (" + rangeIndex+ ") is outside allowable range (0.." + (_list.size()-1) + ")");}return _list.remove(rangeIndex);}
public DimConfig getDimConfig(String dimName) {DimConfig ft = fieldTypes.get(dimName);if (ft == null) {ft = getDefaultDimConfig();}return ft;}
public DescribeStackResourceDriftsResult describeStackResourceDrifts(DescribeStackResourceDriftsRequest request) {request = beforeClientExecution(request);return executeDescribeStackResourceDrifts(request);}
public void setParams(String params) {if (!supportsParams()) {throw new UnsupportedOperationException(getName()+" does not support command line parameters.");}this.params = params;}
public DescribeRepositoryAssociationResult describeRepositoryAssociation(DescribeRepositoryAssociationRequest request) {request = beforeClientExecution(request);return executeDescribeRepositoryAssociation(request);}
public synchronized Enumeration<V> elements() {return new ValueEnumeration();}
public void set(int index, long value) {final int o = index >>> 4;final int b = index & 15;final int shift = b << 2;blocks[o] = (blocks[o] & ~(15L << shift)) | (value << shift);}
public HTMLStripCharFilterFactory(Map<String,String> args) {super(args);escapedTags = getSet(args, "escapedTags");if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public int getEntryPathLength() {return pathLen;}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_option_flag);out.writeShort(field_2_ixals);out.writeShort(field_3_not_used);out.writeByte(field_4_name.length());StringUtil.writeUnicodeStringFlagAndData(out, field_4_name);if(!isOLELink() && !isStdDocumentNameIdentifier()){if(isAutomaticLink()){if(_ddeValues != null) {out.writeByte(_nColumns-1);out.writeShort(_nRows-1);ConstantValueParser.encode(out, _ddeValues);}} else {field_5_name_definition.serialize(out);}}}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[REFRESHALL]\n");buffer.append("    .options      = ").append(HexDump.shortToHex(_options)).append("\n");buffer.append("[/REFRESHALL]\n");return buffer.toString();}
public ContinueDeploymentResult continueDeployment(ContinueDeploymentRequest request) {request = beforeClientExecution(request);return executeContinueDeployment(request);}
public void set(int index, long value) {final int o = index / 3;final int b = index % 3;final int shift = b * 21;blocks[o] = (blocks[o] & ~(2097151L << shift)) | (value << shift);}
public long next() throws IOException {if (ord == valueCount) {throw new EOFException();}if (off == blockSize) {refill();}final long value = values[off++];++ord;return value;}
public static final RevFilter between(Date since, Date until) {return between(since.getTime(), until.getTime());}
public DeleteVaultResult deleteVault(DeleteVaultRequest request) {request = beforeClientExecution(request);return executeDeleteVault(request);}
public final void reset() {it = cachedStates.getStates();}
public void setDetachingSymbolicRef() {detachingSymbolicRef = true;}
public ModifyIdentityIdFormatResult modifyIdentityIdFormat(ModifyIdentityIdFormatRequest request) {request = beforeClientExecution(request);return executeModifyIdentityIdFormat(request);}
public void addException(String word, ArrayList<Object> hyphenatedword) {stoplist.put(word, hyphenatedword);}
public GreekStemFilterFactory(Map<String,String> args) {super(args);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public RegisterTypeResult registerType(RegisterTypeRequest request) {request = beforeClientExecution(request);return executeRegisterType(request);}
public GetAccessControlEffectResult getAccessControlEffect(GetAccessControlEffectRequest request) {request = beforeClientExecution(request);return executeGetAccessControlEffect(request);}
public HSSFShapeGroup createGroup(HSSFChildAnchor anchor) {HSSFShapeGroup group = new HSSFShapeGroup(this, anchor);group.setParent(this);group.setAnchor(anchor);shapes.add(group);onCreate(group);return group;}
public String toExternalString() {final StringBuilder r = new StringBuilder();appendSanitized(r, getName());r.append(" <"); appendSanitized(r, getEmailAddress());r.append("> "); r.append(when / 1000);r.append(' ');appendTimezone(r, tzOffset);return r.toString();}
public static FontCharset valueOf(int value){if(value >= _table.length)return null;return _table[value];}
public NLPSentenceDetectorOp() {sentenceSplitter = null;}
public String resource() {return this.resource;}
public QueryScorer(Query query, String field) {init(query, field, null, true);}
public ActiveTrustedSigners(java.util.List<Signer> items) {setItems(items);}
public final String toString() {StringBuilder sb = new StringBuilder();sb.append(getClass().getName());sb.append(" [");sb.append(formatReferenceAsString());sb.append("]");return sb.toString();}
public UpdateNodegroupConfigResult updateNodegroupConfig(UpdateNodegroupConfigRequest request) {request = beforeClientExecution(request);return executeUpdateNodegroupConfig(request);}
public void fill(int fromIndex, int toIndex, long val) {assert val <= maxValue(getBitsPerValue());assert fromIndex <= toIndex;for (int i = fromIndex; i < toIndex; ++i) {set(i, val);}}
public ListTrainingJobsResult listTrainingJobs(ListTrainingJobsRequest request) {request = beforeClientExecution(request);return executeListTrainingJobs(request);}
public DescribeProfilingGroupResult describeProfilingGroup(DescribeProfilingGroupRequest request) {request = beforeClientExecution(request);return executeDescribeProfilingGroup(request);}
public IgnoreNode(List<FastIgnoreRule> rules) {this.rules = rules;}
public static void fill(char[] array, char value) {for (int i = 0; i < array.length; i++) {array[i] = value;}}
public GetTransitGatewayMulticastDomainAssociationsResult getTransitGatewayMulticastDomainAssociations(GetTransitGatewayMulticastDomainAssociationsRequest request) {request = beforeClientExecution(request);return executeGetTransitGatewayMulticastDomainAssociations(request);}
public LongBuffer compact() {System.arraycopy(backingArray, position + offset, backingArray, offset, remaining());position = limit - position;limit = capacity;mark = UNSET_MARK;return this;}
public GetCelebrityInfoResult getCelebrityInfo(GetCelebrityInfoRequest request) {request = beforeClientExecution(request);return executeGetCelebrityInfo(request);}
public GetTranscriptResult getTranscript(GetTranscriptRequest request) {request = beforeClientExecution(request);return executeGetTranscript(request);}
public DeleteCacheParameterGroupResult deleteCacheParameterGroup(DeleteCacheParameterGroupRequest request) {request = beforeClientExecution(request);return executeDeleteCacheParameterGroup(request);}
public DescribeTagsRequest(java.util.List<Filter> filters) {setFilters(filters);}
public CreateCustomMetadataResult createCustomMetadata(CreateCustomMetadataRequest request) {request = beforeClientExecution(request);return executeCreateCustomMetadata(request);}
public Cluster resumeCluster(ResumeClusterRequest request) {request = beforeClientExecution(request);return executeResumeCluster(request);}
public DescribeMovingAddressesResult describeMovingAddresses(DescribeMovingAddressesRequest request) {request = beforeClientExecution(request);return executeDescribeMovingAddresses(request);}
public SearchAddressBooksResult searchAddressBooks(SearchAddressBooksRequest request) {request = beforeClientExecution(request);return executeSearchAddressBooks(request);}
public UpdateDomainToDomainGroupRequest() {super("Domain", "2018-01-29", "UpdateDomainToDomainGroup");setMethod(MethodType.POST);}
public void add(RevCommit c) {Block b = tail;if (b == null) {b = free.newBlock();b.add(c);head = b;tail = b;return;} else if (b.isFull()) {b = free.newBlock();tail.next = b;tail = b;}b.add(c);}
public FloatBuffer put(int index, float c) {checkIndex(index);byteBuffer.putFloat(index * SizeOf.FLOAT, c);return this;}
public void flush() throws IOException {try {beginWrite();dst.flush();} catch (InterruptedIOException e) {throw writeTimedOut(e);} finally {endWrite();}}
public Set<String> getModified() {return Collections.unmodifiableSet(diff.getModified());}
public LongsRef next(int count) throws IOException {assert count > 0;if (ord == valueCount) {throw new EOFException();}if (off == blockSize) {refill();}count = Math.min(count, blockSize - off);count = (int) Math.min(count, valueCount - ord);valuesRef.offset = off;valuesRef.length = count;off += count;ord += count;return valuesRef;}
public ByteBuffer slice() {return new ReadOnlyHeapByteBuffer(backingArray, remaining(), offset + position);}
public final boolean isEmpty() {return beginA == endA && beginB == endB;}
public static final int commitMessage(byte[] b, int ptr) {final int sz = b.length;if (ptr == 0)ptr += 46; while (ptr < sz && b[ptr] == 'p')ptr += 48; return tagMessage(b, ptr);}
public ValueEval evaluate(ValueEval[] args, OperationEvaluationContext ec) {if (args.length != 2) {return ErrorEval.VALUE_INVALID;}try {double startDateAsNumber = getValue(args[0]);int offsetInMonthAsNumber = (int) getValue(args[1]);Date startDate = DateUtil.getJavaDate(startDateAsNumber);if (startDate == null) {return ErrorEval.VALUE_INVALID;}Calendar calendar = LocaleUtil.getLocaleCalendar();calendar.setTime(startDate);calendar.add(Calendar.MONTH, offsetInMonthAsNumber);return new NumberEval(DateUtil.getExcelDate(calendar.getTime()));} catch (EvaluationException e) {return e.getErrorEval();}}
public DeleteSuggesterResult deleteSuggester(DeleteSuggesterRequest request) {request = beforeClientExecution(request);return executeDeleteSuggester(request);}
public CreatePipelineResult createPipeline(CreatePipelineRequest request) {request = beforeClientExecution(request);return executeCreatePipeline(request);}
public StopDeliveryStreamEncryptionResult stopDeliveryStreamEncryption(StopDeliveryStreamEncryptionRequest request) {request = beforeClientExecution(request);return executeStopDeliveryStreamEncryption(request);}
public DeleteApplicationSnapshotResult deleteApplicationSnapshot(DeleteApplicationSnapshotRequest request) {request = beforeClientExecution(request);return executeDeleteApplicationSnapshot(request);}
public ApplyCommand apply() {return new ApplyCommand(repo);}
public RebootCacheClusterRequest(String cacheClusterId, java.util.List<String> cacheNodeIdsToReboot) {setCacheClusterId(cacheClusterId);setCacheNodeIdsToReboot(cacheNodeIdsToReboot);}
public ModifyCacheClusterRequest(String cacheClusterId) {setCacheClusterId(cacheClusterId);}
public boolean equals(Object obj) {if (this == obj) return true;if (obj == null) return false;if (getClass() != obj.getClass()) return false;ScoreTerm other = (ScoreTerm) obj;if (term == null) {if (other.term != null) return false;} else if (!term.bytesEquals(other.term)) return false;return true;}
public AssociateTransitGatewayMulticastDomainResult associateTransitGatewayMulticastDomain(AssociateTransitGatewayMulticastDomainRequest request) {request = beforeClientExecution(request);return executeAssociateTransitGatewayMulticastDomain(request);}
public UpdateContactResult updateContact(UpdateContactRequest request) {request = beforeClientExecution(request);return executeUpdateContact(request);}
public TableRecord(CellRangeAddress8Bit range) {super(range);field_6_res = 0;}
public CreateProcessingJobResult createProcessingJob(CreateProcessingJobRequest request) {request = beforeClientExecution(request);return executeCreateProcessingJob(request);}
public CharSequence subSequence(int start, int end) {checkStartEndRemaining(start, end);CharSequenceAdapter result = copy(this);result.position = position + start;result.limit = position + end;return result;}
public GetCoipPoolUsageResult getCoipPoolUsage(GetCoipPoolUsageRequest request) {request = beforeClientExecution(request);return executeGetCoipPoolUsage(request);}
public UpdateResolverEndpointResult updateResolverEndpoint(UpdateResolverEndpointRequest request) {request = beforeClientExecution(request);return executeUpdateResolverEndpoint(request);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0) {ValueEval veText;try {veText = OperandResolver.getSingleValue(arg0, srcRowIndex, srcColumnIndex);} catch (EvaluationException e) {return e.getErrorEval();}String strText = OperandResolver.coerceValueToString(veText);Double result = convertTextToNumber(strText);if(result == null) result = parseDateTime(strText);if (result == null) {return ErrorEval.VALUE_INVALID;}return new NumberEval(result.doubleValue());}
public int addExternalName(ExternalNameRecord rec) {ExternalNameRecord[] tmp = new ExternalNameRecord[_externalNameRecords.length + 1];System.arraycopy(_externalNameRecords, 0, tmp, 0, _externalNameRecords.length);tmp[tmp.length - 1] = rec;_externalNameRecords = tmp;return _externalNameRecords.length - 1;}
public DescribePrincipalIdFormatResult describePrincipalIdFormat(DescribePrincipalIdFormatRequest request) {request = beforeClientExecution(request);return executeDescribePrincipalIdFormat(request);}
public ListPartnerEventSourceAccountsResult listPartnerEventSourceAccounts(ListPartnerEventSourceAccountsRequest request) {request = beforeClientExecution(request);return executeListPartnerEventSourceAccounts(request);}
public File getFile() {return file;}
public void onChanged() {if (mSelectedIds.size() > 0) {return;}chooseListToShow();ensureSomeGroupIsExpanded();}
public String getTextAsString() {if (this.text == null)return null;elsereturn this.text.toString();}
public LongBuffer put(long[] src, int srcOffset, int longCount) {Arrays.checkOffsetAndCount(src.length, srcOffset, longCount);if (longCount > remaining()) {throw new BufferOverflowException();}for (int i = srcOffset; i < srcOffset + longCount; ++i) {put(src[i]);}return this;}
@Override public boolean remove(Object object) {synchronized (CopyOnWriteArrayList.this) {int index = indexOf(object);if (index == -1) {return false;}remove(index);return true;}}
public long length() {if (onDiskFile == null) {return super.length();}return onDiskFile.length();}
public FieldBoostMapFCListener(QueryConfigHandler config) {this.config = config;}
public StartActivityStreamResult startActivityStream(StartActivityStreamRequest request) {request = beforeClientExecution(request);return executeStartActivityStream(request);}
public Hyphenation hyphenate(String word, int remainCharCount,int pushCharCount) {char[] w = word.toCharArray();return hyphenate(w, 0, w.length, remainCharCount, pushCharCount);}
public CreateSmsTemplateResult createSmsTemplate(CreateSmsTemplateRequest request) {request = beforeClientExecution(request);return executeCreateSmsTemplate(request);}
public void clear() {int n = mSize;Object[] values = mValues;for (int i = 0; i < n; i++) {values[i] = null;}mSize = 0;mGarbage = false;}
public String toStringTree(Parser parser) {return toString();}
public long get(int index) {final int o = index >>> 2;final int b = index & 3;final int shift = b << 4;return (blocks[o] >>> shift) & 65535L;}
public String toString() {return getType().name() + ": " + getOldId().name() + " "+ getNewId().name() + " " + getRefName();}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval text, ValueEval number_times) {ValueEval veText1;try {veText1 = OperandResolver.getSingleValue(text, srcRowIndex, srcColumnIndex);} catch (EvaluationException e) {return e.getErrorEval();}String strText1 = OperandResolver.coerceValueToString(veText1);double numberOfTime = 0;try {numberOfTime = OperandResolver.coerceValueToDouble(number_times);} catch (EvaluationException e) {return ErrorEval.VALUE_INVALID;}int numberOfTimeInt = (int)numberOfTime;StringBuilder strb = new StringBuilder(strText1.length() * numberOfTimeInt);for(int i = 0; i < numberOfTimeInt; i++) {strb.append(strText1);}if (strb.toString().length() > 32767) {return ErrorEval.VALUE_INVALID;}return new StringEval(strb.toString());}
public Entry<K, V> lastEntry() {return immutableCopy(endpoint(false));}
public DeleteEvaluationResult deleteEvaluation(DeleteEvaluationRequest request) {request = beforeClientExecution(request);return executeDeleteEvaluation(request);}
public ContinueRecord(RecordInputStream in) {_data = in.readRemainder();}
public CreateFilterResult createFilter(CreateFilterRequest request) {request = beforeClientExecution(request);return executeCreateFilter(request);}
public CharSequence subSequence(int start, int end) {checkStartEndRemaining(start, end);CharBuffer result = duplicate();result.limit(position + end);result.position(position + start);return result;}
public CreateTrafficMirrorSessionResult createTrafficMirrorSession(CreateTrafficMirrorSessionRequest request) {request = beforeClientExecution(request);return executeCreateTrafficMirrorSession(request);}
public CreateNodegroupResult createNodegroup(CreateNodegroupRequest request) {request = beforeClientExecution(request);return executeCreateNodegroup(request);}
public SoraniStemFilter create(TokenStream input) {return new SoraniStemFilter(input);}
public UpdateCustomVerificationEmailTemplateResult updateCustomVerificationEmailTemplate(UpdateCustomVerificationEmailTemplateRequest request) {request = beforeClientExecution(request);return executeUpdateCustomVerificationEmailTemplate(request);}
public static FormulaError forInt(int type) throws IllegalArgumentException {FormulaError err = imap.get(type);if(err == null) err = bmap.get((byte)type);if(err == null) throw new IllegalArgumentException("Unknown error type: " + type);return err;}
public DeleteSubnetGroupResult deleteSubnetGroup(DeleteSubnetGroupRequest request) {request = beforeClientExecution(request);return executeDeleteSubnetGroup(request);}
public String toString() {return getClass().getName() + " [" +_error.getString() +"]";}
public Object toObject() {assert exists || 0.0D == value;return exists ? value : null;}
public void destroy() {super.destroy();if (onDiskFile != null) {try {if (!onDiskFile.delete())onDiskFile.deleteOnExit();} finally {onDiskFile = null;}}}
public DecreaseReplicationFactorResult decreaseReplicationFactor(DecreaseReplicationFactorRequest request) {request = beforeClientExecution(request);return executeDecreaseReplicationFactor(request);}
public Counta(){_predicate = defaultPredicate;}
public EvaluationWorkbook getWorkbook() {return _workbook;}
public DescribeRouteTablesResult describeRouteTables() {return describeRouteTables(new DescribeRouteTablesRequest());}
public CreateAssessmentTemplateResult createAssessmentTemplate(CreateAssessmentTemplateRequest request) {request = beforeClientExecution(request);return executeCreateAssessmentTemplate(request);}
public DeleteProjectResult deleteProject(DeleteProjectRequest request) {request = beforeClientExecution(request);return executeDeleteProject(request);}
public DeleteUserPolicyRequest(String userName, String policyName) {setUserName(userName);setPolicyName(policyName);}
public TermVectorsReader clone() {return new CompressingTermVectorsReader(this);}
public void close() {if (sock != null) {try {sch.releaseSession(sock);} finally {sock = null;}}}
public LongBuffer put(long c) {throw new ReadOnlyBufferException();}
public int serialize( int offset, byte[] data ) {LOG.log( DEBUG, "Serializing Workbook with offsets" );int pos = 0;SSTRecord lSST = null;int sstPos = 0;boolean wroteBoundSheets = false;for ( org.apache.poi.hssf.record.Record record : records.getRecords() ) {int len = 0;if (record instanceof SSTRecord) {lSST = (SSTRecord)record;sstPos = pos;}if (record.getSid() == ExtSSTRecord.sid && lSST != null) {record = lSST.createExtSSTRecord(sstPos + offset);}if (record instanceof BoundSheetRecord) {if(!wroteBoundSheets) {for (BoundSheetRecord bsr : boundsheets) {len += bsr.serialize(pos+offset+len, data);}wroteBoundSheets = true;}} else {len = record.serialize( pos + offset, data );}pos += len;}LOG.log( DEBUG, "Exiting serialize workbook" );return pos;}
public DescribeClusterSecurityGroupsResult describeClusterSecurityGroups() {return describeClusterSecurityGroups(new DescribeClusterSecurityGroupsRequest());}
public Explanation explain(Explanation freq, long norm) {return Explanation.match(score(freq.getValue().floatValue(), norm),"score(freq=" + freq.getValue() +"), with freq of:",Collections.singleton(freq));}
public DisassociatePhoneNumberFromUserResult disassociatePhoneNumberFromUser(DisassociatePhoneNumberFromUserRequest request) {request = beforeClientExecution(request);return executeDisassociatePhoneNumberFromUser(request);}
public boolean has(AnyObjectId objectId, int typeHint) throws IOException {try {open(objectId, typeHint);return true;} catch (MissingObjectException notFound) {return false;}}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append("[ATTACHEDLABEL]\n");buffer.append("    .formatFlags          = ").append("0x").append(HexDump.toHex(  getFormatFlags ())).append(" (").append( getFormatFlags() ).append(" )");buffer.append(System.getProperty("line.separator"));buffer.append("         .showActual               = ").append(isShowActual()).append('\n');buffer.append("         .showPercent              = ").append(isShowPercent()).append('\n');buffer.append("         .labelAsPercentage        = ").append(isLabelAsPercentage()).append('\n');buffer.append("         .smoothedLine             = ").append(isSmoothedLine()).append('\n');buffer.append("         .showLabel                = ").append(isShowLabel()).append('\n');buffer.append("         .showBubbleSizes          = ").append(isShowBubbleSizes()).append('\n');buffer.append("[/ATTACHEDLABEL]\n");return buffer.toString();}
public String toString(String field) {StringBuilder buffer = new StringBuilder();buffer.append("spanOr([");Iterator<SpanQuery> i = clauses.iterator();while (i.hasNext()) {SpanQuery clause = i.next();buffer.append(clause.toString(field));if (i.hasNext()) {buffer.append(", ");}}buffer.append("])");return buffer.toString();}
public DisableInsightRulesResult disableInsightRules(DisableInsightRulesRequest request) {request = beforeClientExecution(request);return executeDisableInsightRules(request);}
public BootstrapActionConfig newRunIf(String condition, BootstrapActionConfig config) {List<String> args = config.getScriptBootstrapAction().getArgs();args.add(0, condition);args.add(1, config.getScriptBootstrapAction().getPath());return new BootstrapActionConfig().withName("Run If, " + config.getName()).withScriptBootstrapAction(new ScriptBootstrapActionConfig().withPath("s3:.withArgs(args));}
public final CharBuffer get(char[] dst, int dstOffset, int charCount) {Arrays.checkOffsetAndCount(dst.length, dstOffset, charCount);if (charCount > remaining()) {throw new BufferUnderflowException();}int newPosition = position + charCount;sequence.toString().getChars(position, newPosition, dst, dstOffset);position = newPosition;return this;}
public Set<String> getNames(String section, String subsection) {return getState().getNames(section, subsection);}
public CreateBrokerResult createBroker(CreateBrokerRequest request) {request = beforeClientExecution(request);return executeCreateBroker(request);}
public void onAbsorb(int velocity) {mState = STATE_ABSORB;velocity = Math.max(MIN_VELOCITY, Math.abs(velocity));mStartTime = AnimationUtils.currentAnimationTimeMillis();mDuration = 0.1f + (velocity * 0.03f);mEdgeAlphaStart = 0.f;mEdgeScaleY = mEdgeScaleYStart = 0.f;mGlowAlphaStart = 0.5f;mGlowScaleYStart = 0.f;mEdgeAlphaFinish = Math.max(0, Math.min(velocity * VELOCITY_EDGE_FACTOR, 1));mEdgeScaleYFinish = Math.max(HELD_EDGE_SCALE_Y, Math.min(velocity * VELOCITY_EDGE_FACTOR, 1.f));mGlowScaleYFinish = Math.min(0.025f + (velocity * (velocity / 100) * 0.00015f), 1.75f);mGlowAlphaFinish = Math.max(mGlowAlphaStart, Math.min(velocity * VELOCITY_GLOW_FACTOR * .00001f, MAX_ALPHA));}
public ListSuppressedDestinationsResult listSuppressedDestinations(ListSuppressedDestinationsRequest request) {request = beforeClientExecution(request);return executeListSuppressedDestinations(request);}
public List<Pair<K,V>> getPairs() {List<Pair<K,V>> pairs = new ArrayList<Pair<K,V>>();for (K key : keySet()) {for (V value : get(key)) {pairs.add(new Pair<K,V>(key, value));}}return pairs;}
public void setParams(String params) {super.setParams(params);int k = params.indexOf(",");name = params.substring(0,k).trim();value = params.substring(k+1).trim();}
@Override public V put(K key, V value) {if (!isInBounds(key)) {throw outOfBounds(key, fromBound, toBound);}return putInternal(key, value);}
public DeregisterImageRequest(String imageId) {setImageId(imageId);}
public GetApplicationResult getApplication(GetApplicationRequest request) {request = beforeClientExecution(request);return executeGetApplication(request);}
public DescribeProblemObservationsResult describeProblemObservations(DescribeProblemObservationsRequest request) {request = beforeClientExecution(request);return executeDescribeProblemObservations(request);}
public int fillFields(byte[] data, int offset, EscherRecordFactory recordFactory) {int bytesAfterHeader = readHeader( data, offset );int pos = offset + HEADER_SIZE;System.arraycopy( data, pos, field_1_UID, 0, 16 ); pos += 16;field_2_marker = data[pos]; pos++;setPictureData(data, pos, bytesAfterHeader - 17);return bytesAfterHeader + HEADER_SIZE;}
public static boolean endsWith(BytesRef ref, BytesRef suffix) {int startAt = ref.length - suffix.length;if (startAt < 0) {return false;}return Arrays.equals(ref.bytes, ref.offset + startAt, ref.offset + startAt + suffix.length,suffix.bytes, suffix.offset, suffix.offset + suffix.length);}
public DeleteOptionGroupResult deleteOptionGroup(DeleteOptionGroupRequest request) {request = beforeClientExecution(request);return executeDeleteOptionGroup(request);}
public static String getFromUnicodeLE(byte[] string) {if (string.length == 0) {return "";}return getFromUnicodeLE(string, 0, string.length / 2);}
public CellRangeAddressList() {_list = new ArrayList<>();}
public ValueEval evaluate(ValueEval[] args, OperationEvaluationContext ec) {throw new NotImplementedFunctionException(_functionName);}
public DescribeOptionGroupsResult describeOptionGroups() {return describeOptionGroups(new DescribeOptionGroupsRequest());}
public DisableVpcClassicLinkResult disableVpcClassicLink(DisableVpcClassicLinkRequest request) {request = beforeClientExecution(request);return executeDisableVpcClassicLink(request);}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[SXIDSTM]\n");buffer.append("    .idstm      =").append(HexDump.shortToHex(idstm)).append('\n');buffer.append("[/SXIDSTM]\n");return buffer.toString();}
public ListStackInstancesResult listStackInstances(ListStackInstancesRequest request) {request = beforeClientExecution(request);return executeListStackInstances(request);}
public DescribeCompanyNetworkConfigurationResult describeCompanyNetworkConfiguration(DescribeCompanyNetworkConfigurationRequest request) {request = beforeClientExecution(request);return executeDescribeCompanyNetworkConfiguration(request);}
public final CoderResult flush(CharBuffer out) {if (status != END && status != INIT) {throw new IllegalStateException();}CoderResult result = implFlush(out);if (result == CoderResult.UNDERFLOW) {status = FLUSH;}return result;}
public DescribeDBClustersResult describeDBClusters(DescribeDBClustersRequest request) {request = beforeClientExecution(request);return executeDescribeDBClusters(request);}
public GetDocumentVersionResult getDocumentVersion(GetDocumentVersionRequest request) {request = beforeClientExecution(request);return executeGetDocumentVersion(request);}
public TermData subtract(TermData t1, TermData t2) {if (t2 == NO_OUTPUT) {return t1;}TermData ret;if (statsEqual(t1, t2) && bytesEqual(t1, t2)) {ret = NO_OUTPUT;} else {ret = new TermData(t1.bytes, t1.docFreq, t1.totalTermFreq);}return ret;}
public ModifyCapacityReservationResult modifyCapacityReservation(ModifyCapacityReservationRequest request) {request = beforeClientExecution(request);return executeModifyCapacityReservation(request);}
@Override public int size() {synchronized (mutex) {return c.size();}}
public void decode(byte[] blocks, int blocksOffset, long[] values, int valuesOffset, int iterations) {for (int j = 0; j < iterations; ++j) {values[valuesOffset++] = blocks[blocksOffset++] & 0xFF;}}
public int length() throws UnsupportedOperationException {if (this.type == TYPE_MALFORMED_INPUT || this.type == TYPE_UNMAPPABLE_CHAR) {return this.length;}throw new UnsupportedOperationException("length meaningless for " + toString());}
public String toFormulaString() {throw invalid();}
public E next() {if (iterator.nextIndex() < end) {return iterator.next();}throw new NoSuchElementException();}
public static String toHex(long value) {StringBuilder sb = new StringBuilder(16);writeHex(sb, value, 16, "");return sb.toString();}
public long get(int index) {final int o = index >>> 6;final int b = index & 63;final int shift = b << 0;return (blocks[o] >>> shift) & 1L;}
public int[] clear() {start = end = null;return super.clear();}
public TokenStream init(TokenStream tokenStream) {termAtt = tokenStream.addAttribute(CharTermAttribute.class);return null;}
public UpdateGameServerGroupResult updateGameServerGroup(UpdateGameServerGroupRequest request) {request = beforeClientExecution(request);return executeUpdateGameServerGroup(request);}
public UnmappableCharacterException(int length) {this.inputLength = length;}
public UpdateIdentityProviderConfigurationResult updateIdentityProviderConfiguration(UpdateIdentityProviderConfigurationRequest request) {request = beforeClientExecution(request);return executeUpdateIdentityProviderConfiguration(request);}
@Override public int lastIndexOf(Object object) {Object[] a = array;if (object != null) {for (int i = size - 1; i >= 0; i--) {if (object.equals(a[i])) {return i;}}} else {for (int i = size - 1; i >= 0; i--) {if (a[i] == null) {return i;}}}return -1;}
public ConstantScoreQueryBuilder(QueryBuilderFactory queryFactory) {this.queryFactory = queryFactory;}
public int getNumberOfOnChannelTokens() {int n = 0;fill();for (int i = 0; i < tokens.size(); i++) {Token t = tokens.get(i);if ( t.getChannel()==channel ) n++;if ( t.getType()==Token.EOF ) break;}return n;}
public POIFSDocumentPath(final String [] components)throws IllegalArgumentException{if (components == null){this.components = new String[ 0 ];}else{this.components = new String[ components.length ];for (int j = 0; j < components.length; j++){if ((components[ j ] == null)|| (components[ j ].length() == 0)){throw new IllegalArgumentException("components cannot contain null or empty strings");}this.components[ j ] = components[ j ];}}}
public SQLException(String theReason) {this(theReason, null, 0);}
public ListFragmentsResult listFragments(ListFragmentsRequest request) {request = beforeClientExecution(request);return executeListFragments(request);}
public QueryBuilder getQueryBuilder(String nodeName) {return builders.get(nodeName);}
public CreateDirectoryResult createDirectory(CreateDirectoryRequest request) {request = beforeClientExecution(request);return executeCreateDirectory(request);}
public int getExternalSheetIndex(String workbookName, String sheetName) {return getOrCreateLinkTable().getExternalSheetIndex(workbookName, sheetName, sheetName);}
public V getValue() {return value;}
public K getKey() {return key;}
public boolean hasTransparentBounds() {return transparentBounds;}
public void setKeepEmpty(boolean empty) {keepEmpty = empty;}
public XPathRuleAnywhereElement(String ruleName, int ruleIndex) {super(ruleName);this.ruleIndex = ruleIndex;}
public int getHeight(){return _height;}
public final void write(OpenStringBuilder arr) {write(arr.buf, 0, len);}
public void jumpDrawablesToCurrentState() {super.jumpDrawablesToCurrentState();if (mThumb != null) mThumb.jumpToCurrentState();}
public void setParams(String params) {super.setParams(params);final StreamTokenizer stok = new StreamTokenizer(new StringReader(params));stok.quoteChar('"');stok.quoteChar('\'');stok.eolIsSignificant(false);stok.ordinaryChar(',');try {while (stok.nextToken() != StreamTokenizer.TT_EOF) {switch (stok.ttype) {case ',': {break;}case '\'':case '\"':case StreamTokenizer.TT_WORD: {analyzerNames.add(stok.sval);break;}default: {throw new RuntimeException("Unexpected token: " + stok.toString());}}}} catch (RuntimeException e) {if (e.getMessage().startsWith("Line #")) {throw e;} else {throw new RuntimeException("Line #" + (stok.lineno() + getAlgLineNum()) + ": ", e);}} catch (Throwable t) {throw new RuntimeException("Line #" + (stok.lineno() + getAlgLineNum()) + ": ", t);}}
public DescribeVolumesResult describeVolumes(DescribeVolumesRequest request) {request = beforeClientExecution(request);return executeDescribeVolumes(request);}
public DescribeFlowLogsResult describeFlowLogs(DescribeFlowLogsRequest request) {request = beforeClientExecution(request);return executeDescribeFlowLogs(request);}
public UpdateMethodResult updateMethod(UpdateMethodRequest request) {request = beforeClientExecution(request);return executeUpdateMethod(request);}
public GetAuthorizationTokenRequest() {super("cr", "2016-06-07", "GetAuthorizationToken", "cr");setUriPattern("/tokens");setMethod(MethodType.GET);}
public StopContactResult stopContact(StopContactRequest request) {request = beforeClientExecution(request);return executeStopContact(request);}
public CreateDataSetResult createDataSet(CreateDataSetRequest request) {request = beforeClientExecution(request);return executeCreateDataSet(request);}
public ObjectDatabase newCachedDatabase() {return this;}
public CreateJourneyResult createJourney(CreateJourneyRequest request) {request = beforeClientExecution(request);return executeCreateJourney(request);}
public DeleteDashboardsResult deleteDashboards(DeleteDashboardsRequest request) {request = beforeClientExecution(request);return executeDeleteDashboards(request);}
public UpgradeIndexMergePolicy(MergePolicy in) {super(in);}
public GetHealthCheckCountResult getHealthCheckCount(GetHealthCheckCountRequest request) {request = beforeClientExecution(request);return executeGetHealthCheckCount(request);}
public ChartStartBlockRecord(RecordInputStream in) {rt = in.readShort();grbitFrt = in.readShort();iObjectKind = in.readShort();iObjectContext = in.readShort();iObjectInstance1 = in.readShort();iObjectInstance2 = in.readShort();}
public SeriesRecord(RecordInputStream in) {field_1_categoryDataType = in.readShort();field_2_valuesDataType   = in.readShort();field_3_numCategories    = in.readShort();field_4_numValues        = in.readShort();field_5_bubbleSeriesType = in.readShort();field_6_numBubbleValues  = in.readShort();}
public static Class<? extends CharFilterFactory> lookupClass(String name) {return loader.lookupClass(name);}
public GetPublicKeyResult getPublicKey(GetPublicKeyRequest request) {request = beforeClientExecution(request);return executeGetPublicKey(request);}
public CreateLocalGatewayRouteTableVpcAssociationResult createLocalGatewayRouteTableVpcAssociation(CreateLocalGatewayRouteTableVpcAssociationRequest request) {request = beforeClientExecution(request);return executeCreateLocalGatewayRouteTableVpcAssociation(request);}
public static boolean toBoolean(String stringValue) {if (stringValue == null)throw new NullPointerException(JGitText.get().expectedBooleanStringValue);final Boolean bool = toBooleanOrNull(stringValue);if (bool == null)throw new IllegalArgumentException(MessageFormat.format(JGitText.get().notABoolean, stringValue));return bool.booleanValue();}
public Set<String> getAdded() {return Collections.unmodifiableSet(diff.getAdded());}
public Set<String> getNames(String section) {return getNames(section, null);}
public DescribeCacheClustersResult describeCacheClusters(DescribeCacheClustersRequest request) {request = beforeClientExecution(request);return executeDescribeCacheClusters(request);}
public List<String> getUnmergedPaths() {return unmergedPaths;}
public ValueEval evaluate(ValueEval[] args, OperationEvaluationContext ec) {if (args.length != 2) {return ErrorEval.VALUE_INVALID;}return evaluate(ec.getRowIndex(), ec.getColumnIndex(), args[0], args[1]);}
public int addString(UnicodeString string){field_1_num_strings++;UnicodeString ucs = ( string == null ) ? EMPTY_STRING: string;int rval;int index = field_3_strings.getIndex(ucs);if ( index != -1 ) {rval = index;} else {rval = field_3_strings.size();field_2_num_unique_strings++;SSTDeserializer.addToStringTable( field_3_strings, ucs );}return rval;}
public long getDeltaSearchMemoryLimit() {return deltaSearchMemoryLimit;}
public String toString() {return "Token(\"" + new String(surfaceForm, offset, length) + "\" pos=" + position + " length=" + length +" posLen=" + positionLength + " type=" + type + " wordId=" + wordId +" leftID=" + dictionary.getLeftId(wordId) + ")";}
public String toFormulaString(FormulaRenderingWorkbook book) {return ExternSheetNameResolver.prependSheetName(book, field_1_index_extern_sheet, formatReferenceAsString());}
public E get(int index) {return (E) elements[index];}
public byte[] getCachedBytes() {return data;}
public DescribeConnectionsResult describeConnections() {return describeConnections(new DescribeConnectionsRequest());}
public void ensureCapacity(int minimumCapacity) {Object[] a = array;if (a.length < minimumCapacity) {Object[] newArray = new Object[minimumCapacity];System.arraycopy(a, 0, newArray, 0, size);array = newArray;modCount++;}}
public DeleteLifecycleHookResult deleteLifecycleHook(DeleteLifecycleHookRequest request) {request = beforeClientExecution(request);return executeDeleteLifecycleHook(request);}
public final float maxBytesPerChar() {return maxBytesPerChar;}
public BlankCellRectangleGroup(int firstRowIndex, int firstColumnIndex, int lastColumnIndex) {_firstRowIndex = firstRowIndex;_firstColumnIndex = firstColumnIndex;_lastColumnIndex = lastColumnIndex;_lastRowIndex = firstRowIndex;}
public int findEndOfRowOutlineGroup(int row) {int level = getRow( row ).getOutlineLevel();int currentRow;for (currentRow = row; currentRow < getLastRowNum(); currentRow++) {if (getRow(currentRow) == null || getRow(currentRow).getOutlineLevel() < level) {break;}}return currentRow-1;}
public String getEncoding() {if (encoder == null) {return null;}return HistoricalCharsetNames.get(encoder.charset());}
public void clearAllCachedResultValues() {_cache.clear();_sheetIndexesBySheet.clear();_workbook.clearAllCachedResultValues();}
public final String toString() {StringBuilder sb = new StringBuilder();String recordName = getRecordName();sb.append("[").append(recordName).append("]\n");sb.append("    .row    = ").append(HexDump.shortToHex(getRow())).append("\n");sb.append("    .col    = ").append(HexDump.shortToHex(getColumn())).append("\n");if (isBiff2()) {sb.append("    .cellattrs = ").append(HexDump.shortToHex(getCellAttrs())).append("\n");} else {sb.append("    .xfindex   = ").append(HexDump.shortToHex(getXFIndex())).append("\n");}appendValueText(sb);sb.append("\n");sb.append("[/").append(recordName).append("]\n");return sb.toString();}
public DescribeDBClusterEndpointsResult describeDBClusterEndpoints(DescribeDBClusterEndpointsRequest request) {request = beforeClientExecution(request);return executeDescribeDBClusterEndpoints(request);}
public boolean renameTo(final String newName){boolean rval = false;if (!isRoot()){rval = _parent.changeName(getName(), newName);}return rval;}
public Explanation explain(Explanation freq, long norm) {List<Explanation> subs = new ArrayList<>();for (SimScorer subScorer : subScorers) {subs.add(subScorer.explain(freq, norm));}return Explanation.match(score(freq.getValue().floatValue(), norm), "sum of:", subs);}
public DocTermsIndexDocValues(ValueSource vs, LeafReaderContext context, String field) throws IOException {this(vs, open(context, field));}
public static int compareTo(Ref o1, Ref o2) {return o1.getName().compareTo(o2.getName());}
public Dimension getImageDimension(){InternalWorkbook iwb = getPatriarch().getSheet().getWorkbook().getWorkbook();EscherBSERecord bse = iwb.getBSERecord(getPictureIndex());byte[] data = bse.getBlipRecord().getPicturedata();int type = bse.getBlipTypeWin32();return ImageUtils.getImageDimension(new ByteArrayInputStream(data), type);}
public static double var(double[] v) {double r = Double.NaN;if (v!=null && v.length > 1) {r = devsq(v) / (v.length - 1);}return r;}
public UpdateCloudFrontOriginAccessIdentityRequest(CloudFrontOriginAccessIdentityConfig cloudFrontOriginAccessIdentityConfig, String id, String ifMatch) {setCloudFrontOriginAccessIdentityConfig(cloudFrontOriginAccessIdentityConfig);setId(id);setIfMatch(ifMatch);}
public DiffCommand setDestinationPrefix(String destinationPrefix) {this.destinationPrefix = destinationPrefix;return this;}
public int available() throws IOException {return IoBridge.available(fd);}
final public SrndQuery NotQuery() throws ParseException {SrndQuery q;ArrayList<SrndQuery> queries = null;Token oprt = null;q = NQuery();label_4:while (true) {switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {case NOT:;break;default:jj_la1[2] = jj_gen;break label_4;}oprt = jj_consume_token(NOT);if (queries == null) {queries = new ArrayList<SrndQuery>();queries.add(q);}q = NQuery();queries.add(q);}{if (true) return (queries == null) ? q : getNotQuery(queries, oprt);}throw new Error("Missing return statement in function");}
public String toString() {StringBuilder sb = new StringBuilder();sb.append('[').append("USERSVIEWEND").append("] (0x");sb.append(Integer.toHexString(sid).toUpperCase(Locale.ROOT)).append(")\n");sb.append("  rawData=").append(HexDump.toHex(_rawData)).append("\n");sb.append("[/").append("USERSVIEWEND").append("]\n");return sb.toString();}
public FloatBuffer asReadOnlyBuffer() {FloatToByteBufferAdapter buf = new FloatToByteBufferAdapter(byteBuffer.asReadOnlyBuffer());buf.limit = limit;buf.position = position;buf.mark = mark;buf.byteBuffer.order = byteBuffer.order;return buf;}
public LogCommand log() {return new LogCommand(repo);}
public CreateDomainResult createDomain(CreateDomainRequest request) {request = beforeClientExecution(request);return executeCreateDomain(request);}
public int getWeight() {return WEIGHT_UNKNOWN;}
public ChartStartObjectRecord(RecordInputStream in) {rt = in.readShort();grbitFrt = in.readShort();iObjectKind = in.readShort();iObjectContext = in.readShort();iObjectInstance1 = in.readShort();iObjectInstance2 = in.readShort();}
public void remove() {if (lastReturned == null)throw new IllegalStateException();ConcurrentHashMap.this.remove(lastReturned.key);lastReturned = null;}
public DescribeMetricCollectionTypesResult describeMetricCollectionTypes(DescribeMetricCollectionTypesRequest request) {request = beforeClientExecution(request);return executeDescribeMetricCollectionTypes(request);}
public UpdateFieldLevelEncryptionProfileResult updateFieldLevelEncryptionProfile(UpdateFieldLevelEncryptionProfileRequest request) {request = beforeClientExecution(request);return executeUpdateFieldLevelEncryptionProfile(request);}
public Ref getLeaf() {return this;}
public int lastIndexOf(Object object) {if (object != null) {for (int i = a.length - 1; i >= 0; i--) {if (object.equals(a[i])) {return i;}}} else {for (int i = a.length - 1; i >= 0; i--) {if (a[i] == null) {return i;}}}return -1;}
public DefaultBulkScorer(Scorer scorer) {if (scorer == null) {throw new NullPointerException();}this.scorer = scorer;this.iterator = scorer.iterator();this.twoPhase = scorer.twoPhaseIterator();}
public CreateRepoAuthorizationRequest() {super("cr", "2016-06-07", "CreateRepoAuthorization", "cr");setUriPattern("/repos/[RepoNamespace]/[RepoName]/authorizations");setMethod(MethodType.PUT);}
public TokenStream create(TokenStream input) {return new PortugueseLightStemFilter(input);}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[TABLESTYLES]\n");buffer.append("    .rt      =").append(HexDump.shortToHex(rt)).append('\n');buffer.append("    .grbitFrt=").append(HexDump.shortToHex(grbitFrt)).append('\n');buffer.append("    .unused  =").append(HexDump.toHex(unused)).append('\n');buffer.append("    .cts=").append(HexDump.intToHex(cts)).append('\n');buffer.append("    .rgchDefListStyle=").append(rgchDefListStyle).append('\n');buffer.append("    .rgchDefPivotStyle=").append(rgchDefPivotStyle).append('\n');buffer.append("[/TABLESTYLES]\n");return buffer.toString();}
public synchronized Enumeration<K> keys() {return new KeyEnumeration();}
public DescribeInstanceTypesResult describeInstanceTypes(DescribeInstanceTypesRequest request) {request = beforeClientExecution(request);return executeDescribeInstanceTypes(request);}
public RefUpdate.Result getResult() {return rc;}
public UpdateBasePathMappingResult updateBasePathMapping(UpdateBasePathMappingRequest request) {request = beforeClientExecution(request);return executeUpdateBasePathMapping(request);}
public UpdateDocumentResult updateDocument(UpdateDocumentRequest request) {request = beforeClientExecution(request);return executeUpdateDocument(request);}
public void setStreamFileThreshold(int newLimit) {streamFileThreshold = newLimit;}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[EXTSST]\n");buffer.append("    .dsst           = ").append(Integer.toHexString(_stringsPerBucket)).append("\n");buffer.append("    .numInfoRecords = ").append(_sstInfos.length).append("\n");for (int k = 0; k < _sstInfos.length; k++){buffer.append("    .inforecord     = ").append(k).append("\n");buffer.append("    .streampos      = ").append(Integer.toHexString(_sstInfos[k].getStreamPos())).append("\n");buffer.append("    .sstoffset      = ").append(Integer.toHexString(_sstInfos[k].getBucketSSTOffset())).append("\n");}buffer.append("[/EXTSST]\n");return buffer.toString();}
public void setCRC(int crc) {this.crc = crc;}
public RevFilter getRevFilter() {return filter;}
public SrndPrefixQuery(String prefix, boolean quoted, char truncator) {super(quoted);this.prefix = prefix;prefixRef = new BytesRef(prefix);this.truncator = truncator;}
public byte readByte() throws IOException {int v = is.read();if (v == -1) throw new EOFException();return (byte) v;}
public GetWorkGroupResult getWorkGroup(GetWorkGroupRequest request) {request = beforeClientExecution(request);return executeGetWorkGroup(request);}
public PutBlockPublicAccessConfigurationResult putBlockPublicAccessConfiguration(PutBlockPublicAccessConfigurationRequest request) {request = beforeClientExecution(request);return executePutBlockPublicAccessConfiguration(request);}
public String toString() {final StringBuilder r = new StringBuilder();r.append('[');for (int i = 0; i < count; i++) {if (i > 0)r.append(", "); r.append(entries[i]);}r.append(']');return r.toString();}
public int get(int index) {checkIndex(index);return byteBuffer.getInt(index * SizeOf.INT);}
public CreateAlbumRequest() {super("CloudPhoto", "2017-07-11", "CreateAlbum", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public FileTreeIterator(File root, FS fs, WorkingTreeOptions options) {this(root, fs, options, DefaultFileModeStrategy.INSTANCE);}
public int byteAt(int idx) {return bytes[idx].value;}
public DescribeTypeRegistrationResult describeTypeRegistration(DescribeTypeRegistrationRequest request) {request = beforeClientExecution(request);return executeDescribeTypeRegistration(request);}
public TerminateInstancesResult terminateInstances(TerminateInstancesRequest request) {request = beforeClientExecution(request);return executeTerminateInstances(request);}
public DoubleBuffer duplicate() {ByteBuffer bb = byteBuffer.duplicate().order(byteBuffer.order());DoubleToByteBufferAdapter buf = new DoubleToByteBufferAdapter(bb);buf.limit = limit;buf.position = position;buf.mark = mark;return buf;}
public OR(SemanticContext a, SemanticContext b) {Set<SemanticContext> operands = new HashSet<SemanticContext>();if ( a instanceof OR ) operands.addAll(Arrays.asList(((OR)a).opnds));else operands.add(a);if ( b instanceof OR ) operands.addAll(Arrays.asList(((OR)b).opnds));else operands.add(b);List<PrecedencePredicate> precedencePredicates = filterPrecedencePredicates(operands);if (!precedencePredicates.isEmpty()) {PrecedencePredicate reduced = Collections.max(precedencePredicates);operands.add(reduced);}this.opnds = operands.toArray(new SemanticContext[operands.size()]);}
public void serialize(LittleEndianOutput out) {out.writeShort(_formats.length);for(int i=0; i<_formats.length; i++){_formats[i].serialize(out);}}
public DescribeAvailabilityOptionsResult describeAvailabilityOptions(DescribeAvailabilityOptionsRequest request) {request = beforeClientExecution(request);return executeDescribeAvailabilityOptions(request);}
public int getOffset() {return offset;}
public static float[] grow(float[] array) {return grow(array, 1 + array.length);}
public ListMetricsResult listMetrics() {return listMetrics(new ListMetricsRequest());}
public int findFirstRecordLocBySid(short sid) {int index = 0;for (org.apache.poi.hssf.record.Record record : records.getRecords() ) {if (record.getSid() == sid) {return index;}index ++;}return -1;}
public DeleteVpnConnectionRouteResult deleteVpnConnectionRoute(DeleteVpnConnectionRouteRequest request) {request = beforeClientExecution(request);return executeDeleteVpnConnectionRoute(request);}
