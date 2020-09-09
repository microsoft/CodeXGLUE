public void serialize(LittleEndianOutput out) {out.writeShort(field_1_vcenter);}
public void addAll(BlockList<T> src) {if (src.size == 0)return;int srcDirIdx = 0;for (; srcDirIdx < src.tailDirIdx; srcDirIdx++)addAll(src.directory[srcDirIdx], 0, BLOCK_SIZE);if (src.tailBlkIdx != 0)addAll(src.tailBlock, 0, src.tailBlkIdx);}
public void writeByte(byte b) {if (upto == blockSize) {if (currentBlock != null) {addBlock(currentBlock);}currentBlock = new byte[blockSize];upto = 0;}currentBlock[upto++] = b;}
public ObjectId getObjectId() {return objectId;}
public DeleteDomainEntryResult deleteDomainEntry(DeleteDomainEntryRequest request) {request = beforeClientExecution(request);return executeDeleteDomainEntry(request);}
public long ramBytesUsed() {return ((termOffsets!=null)? termOffsets.ramBytesUsed() : 0) +((termsDictOffsets!=null)? termsDictOffsets.ramBytesUsed() : 0);}
public final String getFullMessage() {byte[] raw = buffer;int msgB = RawParseUtils.tagMessage(raw, 0);if (msgB < 0) {return ""; }return RawParseUtils.decode(guessEncoding(), raw, msgB, raw.length);}
public POIFSFileSystem() {this(true);_header.setBATCount(1);_header.setBATArray(new int[]{1});BATBlock bb = BATBlock.createEmptyBATBlock(bigBlockSize, false);bb.setOurBlockIndex(1);_bat_blocks.add(bb);setNextBlock(0, POIFSConstants.END_OF_CHAIN);setNextBlock(1, POIFSConstants.FAT_SECTOR_BLOCK);_property_table.setStartBlock(0);}
public void init(int address) {slice = pool.buffers[address >> ByteBlockPool.BYTE_BLOCK_SHIFT];assert slice != null;upto = address & ByteBlockPool.BYTE_BLOCK_MASK;offset0 = address;assert upto < slice.length;}
public SubmoduleAddCommand setPath(String path) {this.path = path;return this;}
public ListIngestionsResult listIngestions(ListIngestionsRequest request) {request = beforeClientExecution(request);return executeListIngestions(request);}
public QueryParserTokenManager(CharStream stream, int lexState){this(stream);SwitchTo(lexState);}
public GetShardIteratorResult getShardIterator(GetShardIteratorRequest request) {request = beforeClientExecution(request);return executeGetShardIterator(request);}
public ModifyStrategyRequest() {super("aegis", "2016-11-11", "ModifyStrategy", "vipaegis");setMethod(MethodType.POST);}
public boolean ready() throws IOException {synchronized (lock) {if (in == null) {throw new IOException("InputStreamReader is closed");}try {return bytes.hasRemaining() || in.available() > 0;} catch (IOException e) {return false;}}}
public EscherOptRecord getOptRecord() {return _optRecord;}
public synchronized int read(byte[] buffer, int offset, int length) {if (buffer == null) {throw new NullPointerException("buffer == null");}Arrays.checkOffsetAndCount(buffer.length, offset, length);if (length == 0) {return 0;}int copylen = count - pos < length ? count - pos : length;for (int i = 0; i < copylen; i++) {buffer[offset + i] = (byte) this.buffer.charAt(pos + i);}pos += copylen;return copylen;}
public OpenNLPSentenceBreakIterator(NLPSentenceDetectorOp sentenceOp) {this.sentenceOp = sentenceOp;}
public void print(String str) {write(str != null ? str : String.valueOf((Object) null));}
public NotImplementedFunctionException(String functionName, NotImplementedException cause) {super(functionName, cause);this.functionName = functionName;}
public V next() {return super.nextEntry().getValue();}
public final void readBytes(byte[] b, int offset, int len, boolean useBuffer) throws IOException {int available = bufferLength - bufferPosition;if(len <= available){if(len>0) System.arraycopy(buffer, bufferPosition, b, offset, len);bufferPosition+=len;} else {if(available > 0){System.arraycopy(buffer, bufferPosition, b, offset, available);offset += available;len -= available;bufferPosition += available;}if (useBuffer && len<bufferSize){refill();if(bufferLength<len){System.arraycopy(buffer, 0, b, offset, bufferLength);throw new EOFException("read past EOF: " + this);} else {System.arraycopy(buffer, 0, b, offset, len);bufferPosition=len;}} else {long after = bufferStart+bufferPosition+len;if(after > length())throw new EOFException("read past EOF: " + this);readInternal(b, offset, len);bufferStart = after;bufferPosition = 0;bufferLength = 0;                    }}}
public TagQueueResult tagQueue(TagQueueRequest request) {request = beforeClientExecution(request);return executeTagQueue(request);}
public void remove() {throw new UnsupportedOperationException();}
public CacheSubnetGroup modifyCacheSubnetGroup(ModifyCacheSubnetGroupRequest request) {request = beforeClientExecution(request);return executeModifyCacheSubnetGroup(request);}
public void setParams(String params) {super.setParams(params);language = country = variant = "";StringTokenizer st = new StringTokenizer(params, ",");if (st.hasMoreTokens())language = st.nextToken();if (st.hasMoreTokens())country = st.nextToken();if (st.hasMoreTokens())variant = st.nextToken();}
public DeleteDocumentationVersionResult deleteDocumentationVersion(DeleteDocumentationVersionRequest request) {request = beforeClientExecution(request);return executeDeleteDocumentationVersion(request);}
public boolean equals(Object obj) {if (!(obj instanceof FacetLabel)) {return false;}FacetLabel other = (FacetLabel) obj;if (length != other.length) {return false; }for (int i = length - 1; i >= 0; i--) {if (!components[i].equals(other.components[i])) {return false;}}return true;}
public GetInstanceAccessDetailsResult getInstanceAccessDetails(GetInstanceAccessDetailsRequest request) {request = beforeClientExecution(request);return executeGetInstanceAccessDetails(request);}
public HSSFPolygon createPolygon(HSSFChildAnchor anchor) {HSSFPolygon shape = new HSSFPolygon(this, anchor);shape.setParent(this);shape.setAnchor(anchor);shapes.add(shape);onCreate(shape);return shape;}
public String getSheetName(int sheetIndex) {return getBoundSheetRec(sheetIndex).getSheetname();}
public GetDashboardResult getDashboard(GetDashboardRequest request) {request = beforeClientExecution(request);return executeGetDashboard(request);}
public AssociateSigninDelegateGroupsWithAccountResult associateSigninDelegateGroupsWithAccount(AssociateSigninDelegateGroupsWithAccountRequest request) {request = beforeClientExecution(request);return executeAssociateSigninDelegateGroupsWithAccount(request);}
public void addMultipleBlanks(MulBlankRecord mbr) {for (int j = 0; j < mbr.getNumColumns(); j++) {BlankRecord br = new BlankRecord();br.setColumn(( short ) (j + mbr.getFirstColumn()));br.setRow(mbr.getRow());br.setXFIndex(mbr.getXFAt(j));insertCell(br);}}
public static String quote(String string) {StringBuilder sb = new StringBuilder();sb.append("\\Q");int apos = 0;int k;while ((k = string.indexOf("\\E", apos)) >= 0) {sb.append(string.substring(apos, k + 2)).append("\\\\E\\Q");apos = k + 2;}return sb.append(string.substring(apos)).append("\\E").toString();}
public ByteBuffer putInt(int value) {throw new ReadOnlyBufferException();}
public ArrayPtg(Object[][] values2d) {int nColumns = values2d[0].length;int nRows = values2d.length;_nColumns = (short) nColumns;_nRows = (short) nRows;Object[] vv = new Object[_nColumns * _nRows];for (int r=0; r<nRows; r++) {Object[] rowData = values2d[r];for (int c=0; c<nColumns; c++) {vv[getValueIndex(c, r)] = rowData[c];}}_arrayValues = vv;_reserved0Int = 0;_reserved1Short = 0;_reserved2Byte = 0;}
public GetIceServerConfigResult getIceServerConfig(GetIceServerConfigRequest request) {request = beforeClientExecution(request);return executeGetIceServerConfig(request);}
public String toString() {return getClass().getName() + " [" +getValueAsString() +"]";}
public String toString(String field) {return "ToChildBlockJoinQuery ("+parentQuery.toString()+")";}
public final void incRef() {refCount.incrementAndGet();}
public UpdateConfigurationSetSendingEnabledResult updateConfigurationSetSendingEnabled(UpdateConfigurationSetSendingEnabledRequest request) {request = beforeClientExecution(request);return executeUpdateConfigurationSetSendingEnabled(request);}
public int getNextXBATChainOffset() {return getXBATEntriesPerBlock() * LittleEndianConsts.INT_SIZE;}
public void multiplyByPowerOfTen(int pow10) {TenPower tp = TenPower.getInstance(Math.abs(pow10));if (pow10 < 0) {mulShift(tp._divisor, tp._divisorShift);} else {mulShift(tp._multiplicand, tp._multiplierShift);}}
public String toString(){final StringBuilder b = new StringBuilder();final int          l = length();b.append(File.separatorChar);for (int i = 0; i < l; i++){b.append(getComponent(i));if (i < l - 1){b.append(File.separatorChar);}}return b.toString();}
public InstanceProfileCredentialsProvider withFetcher(ECSMetadataServiceCredentialsFetcher fetcher) {this.fetcher = fetcher;this.fetcher.setRoleName(roleName);return this;}
public void setProgressMonitor(ProgressMonitor pm) {progressMonitor = pm;}
public void reset() {if (!first()) {ptr = 0;if (!eof())parseEntry();}}
public E previous() {if (iterator.previousIndex() >= start) {return iterator.previous();}throw new NoSuchElementException();}
public String getNewPrefix() {return this.newPrefix;}
public int indexOfValue(int value) {for (int i = 0; i < mSize; i++)if (mValues[i] == value)return i;return -1;}
public List<CharsRef> uniqueStems(char word[], int length) {List<CharsRef> stems = stem(word, length);if (stems.size() < 2) {return stems;}CharArraySet terms = new CharArraySet(8, dictionary.ignoreCase);List<CharsRef> deduped = new ArrayList<>();for (CharsRef s : stems) {if (!terms.contains(s)) {deduped.add(s);terms.add(s);}}return deduped;}
public GetGatewayResponsesResult getGatewayResponses(GetGatewayResponsesRequest request) {request = beforeClientExecution(request);return executeGetGatewayResponses(request);}
public void setPosition(long pos) {currentBlockIndex = (int) (pos >> blockBits);currentBlock = blocks[currentBlockIndex];currentBlockUpto = (int) (pos & blockMask);}
public long skip(long n) {int s = (int) Math.min(available(), Math.max(0, n));ptr += s;return s;}
public BootstrapActionDetail(BootstrapActionConfig bootstrapActionConfig) {setBootstrapActionConfig(bootstrapActionConfig);}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_row);out.writeShort(field_2_col);out.writeShort(field_3_flags);out.writeShort(field_4_shapeid);out.writeShort(field_6_author.length());out.writeByte(field_5_hasMultibyte ? 0x01 : 0x00);if (field_5_hasMultibyte) {StringUtil.putUnicodeLE(field_6_author, out);} else {StringUtil.putCompressedUnicode(field_6_author, out);}if (field_7_padding != null) {out.writeByte(field_7_padding.intValue());}}
public int lastIndexOf(String string) {return lastIndexOf(string, count);}
public boolean add(E object) {return addLastImpl(object);}
public void unsetSection(String section, String subsection) {ConfigSnapshot src, res;do {src = state.get();res = unsetSection(src, section, subsection);} while (!state.compareAndSet(src, res));}
public final String getTagName() {return tagName;}
public void addSubRecord(int index, SubRecord element) {subrecords.add(index, element);}
public boolean remove(Object o) {synchronized (mutex) {return delegate().remove(o);}}
public DoubleMetaphoneFilter create(TokenStream input) {return new DoubleMetaphoneFilter(input, maxCodeLength, inject);}
public long length() {return inCoreLength();}
public void setValue(boolean newValue) {value = newValue;}
public Pair(ContentSource oldSource, ContentSource newSource) {this.oldSource = oldSource;this.newSource = newSource;}
public int get(int i) {if (count <= i)throw new ArrayIndexOutOfBoundsException(i);return entries[i];}
public CreateRepoRequest() {super("cr", "2016-06-07", "CreateRepo", "cr");setUriPattern("/repos");setMethod(MethodType.PUT);}
public boolean isDeltaBaseAsOffset() {return deltaBaseAsOffset;}
public void remove() {if (expectedModCount == list.modCount) {if (lastLink != null) {Link<ET> next = lastLink.next;Link<ET> previous = lastLink.previous;next.previous = previous;previous.next = next;if (lastLink == link) {pos--;}link = previous;lastLink = null;expectedModCount++;list.size--;list.modCount++;} else {throw new IllegalStateException();}} else {throw new ConcurrentModificationException();}}
public MergeShardsResult mergeShards(MergeShardsRequest request) {request = beforeClientExecution(request);return executeMergeShards(request);}
public AllocateHostedConnectionResult allocateHostedConnection(AllocateHostedConnectionRequest request) {request = beforeClientExecution(request);return executeAllocateHostedConnection(request);}
public int getBeginIndex() {return start;}
public static final WeightedTerm[] getTerms(Query query){return getTerms(query,false);}
public ByteBuffer compact() {throw new ReadOnlyBufferException();}
public void decode(byte[] blocks, int blocksOffset, long[] values, int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final long byte0 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = byte0 >>> 2;final long byte1 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte0 & 3) << 4) | (byte1 >>> 4);final long byte2 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte1 & 15) << 2) | (byte2 >>> 6);values[valuesOffset++] = byte2 & 63;}}
public String getHumanishName() throws IllegalArgumentException {String s = getPath();if ("/".equals(s) || "".equals(s)) s = getHost();if (s == null) throw new IllegalArgumentException();String[] elements;if ("file".equals(scheme) || LOCAL_FILE.matcher(s).matches()) elements = s.split("[\\" + File.separatorChar + "/]"); elseelements = s.split("/+"); if (elements.length == 0)throw new IllegalArgumentException();String result = elements[elements.length - 1];if (Constants.DOT_GIT.equals(result))result = elements[elements.length - 2];else if (result.endsWith(Constants.DOT_GIT_EXT))result = result.substring(0, result.length()- Constants.DOT_GIT_EXT.length());return result;}
public DescribeNotebookInstanceLifecycleConfigResult describeNotebookInstanceLifecycleConfig(DescribeNotebookInstanceLifecycleConfigRequest request) {request = beforeClientExecution(request);return executeDescribeNotebookInstanceLifecycleConfig(request);}
public String getAccessKeySecret() {return this.accessKeySecret;}
public CreateVpnConnectionResult createVpnConnection(CreateVpnConnectionRequest request) {request = beforeClientExecution(request);return executeCreateVpnConnection(request);}
public DescribeVoicesResult describeVoices(DescribeVoicesRequest request) {request = beforeClientExecution(request);return executeDescribeVoices(request);}
public ListMonitoringExecutionsResult listMonitoringExecutions(ListMonitoringExecutionsRequest request) {request = beforeClientExecution(request);return executeListMonitoringExecutions(request);}
public DescribeJobRequest(String vaultName, String jobId) {setVaultName(vaultName);setJobId(jobId);}
public EscherRecord getEscherRecord(int index){return escherRecords.get(index);}
public GetApisResult getApis(GetApisRequest request) {request = beforeClientExecution(request);return executeGetApis(request);}
public DeleteSmsChannelResult deleteSmsChannel(DeleteSmsChannelRequest request) {request = beforeClientExecution(request);return executeDeleteSmsChannel(request);}
public TrackingRefUpdate getTrackingRefUpdate() {return trackingRefUpdate;}
public void print(boolean b) {print(String.valueOf(b));}
public QueryNode getChild() {return getChildren().get(0);}
public NotIgnoredFilter(int workdirTreeIndex) {this.index = workdirTreeIndex;}
public AreaRecord(RecordInputStream in) {field_1_formatFlags            = in.readShort();}
public GetThumbnailRequest() {super("CloudPhoto", "2017-07-11", "GetThumbnail", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public DescribeTransitGatewayVpcAttachmentsResult describeTransitGatewayVpcAttachments(DescribeTransitGatewayVpcAttachmentsRequest request) {request = beforeClientExecution(request);return executeDescribeTransitGatewayVpcAttachments(request);}
public PutVoiceConnectorStreamingConfigurationResult putVoiceConnectorStreamingConfiguration(PutVoiceConnectorStreamingConfigurationRequest request) {request = beforeClientExecution(request);return executePutVoiceConnectorStreamingConfiguration(request);}
public OrdRange getOrdRange(String dim) {return prefixToOrdRange.get(dim);}
public String toString() {String symbol = "";if (startIndex >= 0 && startIndex < getInputStream().size()) {symbol = getInputStream().getText(Interval.of(startIndex,startIndex));symbol = Utils.escapeWhitespace(symbol, false);}return String.format(Locale.getDefault(), "%s('%s')", LexerNoViableAltException.class.getSimpleName(), symbol);}
public E peek() {return peekFirstImpl();}
public CreateWorkspacesResult createWorkspaces(CreateWorkspacesRequest request) {request = beforeClientExecution(request);return executeCreateWorkspaces(request);}
public NumberFormatIndexRecord clone() {return copy();}
public DescribeRepositoriesResult describeRepositories(DescribeRepositoriesRequest request) {request = beforeClientExecution(request);return executeDescribeRepositories(request);}
public SparseIntArray(int initialCapacity) {initialCapacity = ArrayUtils.idealIntArraySize(initialCapacity);mKeys = new int[initialCapacity];mValues = new int[initialCapacity];mSize = 0;}
public HyphenatedWordsFilter create(TokenStream input) {return new HyphenatedWordsFilter(input);}
public CreateDistributionWithTagsResult createDistributionWithTags(CreateDistributionWithTagsRequest request) {request = beforeClientExecution(request);return executeCreateDistributionWithTags(request);}
public RandomAccessFile(String fileName, String mode) throws FileNotFoundException {this(new File(fileName), mode);}
public DeleteWorkspaceImageResult deleteWorkspaceImage(DeleteWorkspaceImageRequest request) {request = beforeClientExecution(request);return executeDeleteWorkspaceImage(request);}
public static String toHex(long value) {StringBuilder sb = new StringBuilder(16);writeHex(sb, value, 16, "");return sb.toString();}
public UpdateDistributionResult updateDistribution(UpdateDistributionRequest request) {request = beforeClientExecution(request);return executeUpdateDistribution(request);}
public HSSFColor getColor(short index){if (index == HSSFColorPredefined.AUTOMATIC.getIndex()) {return HSSFColorPredefined.AUTOMATIC.getColor();}byte[] b = _palette.getColor(index);return (b == null) ? null : new CustomColor(index, b);}
public ValueEval evaluate(ValueEval[] operands, int srcRow, int srcCol) {throw new NotImplementedFunctionException(_functionName);}
public void serialize(LittleEndianOutput out) {out.writeShort((short)field_1_number_crn_records);out.writeShort((short)field_2_sheet_table_index);}
public DescribeDBEngineVersionsResult describeDBEngineVersions() {return describeDBEngineVersions(new DescribeDBEngineVersionsRequest());}
public FormatRun(short character, short fontIndex) {this._character = character;this._fontIndex = fontIndex;}
public static byte[] toBigEndianUtf16Bytes(char[] chars, int offset, int length) {byte[] result = new byte[length * 2];int end = offset + length;int resultIndex = 0;for (int i = offset; i < end; ++i) {char ch = chars[i];result[resultIndex++] = (byte) (ch >> 8);result[resultIndex++] = (byte) ch;}return result;}
public UploadArchiveResult uploadArchive(UploadArchiveRequest request) {request = beforeClientExecution(request);return executeUploadArchive(request);}
public List<Token> getHiddenTokensToLeft(int tokenIndex) {return getHiddenTokensToLeft(tokenIndex, -1);}
public boolean equals(Object obj) {if (this == obj)return true;if (!super.equals(obj))return false;if (getClass() != obj.getClass())return false;AutomatonQuery other = (AutomatonQuery) obj;if (!compiled.equals(other.compiled))return false;if (term == null) {if (other.term != null)return false;} else if (!term.equals(other.term))return false;return true;}
public SpanQuery makeSpanClause() {SpanQuery [] spanQueries = new SpanQuery[size()];Iterator<SpanQuery> sqi = weightBySpanQuery.keySet().iterator();int i = 0;while (sqi.hasNext()) {SpanQuery sq = sqi.next();float boost = weightBySpanQuery.get(sq);if (boost != 1f) {sq = new SpanBoostQuery(sq, boost);}spanQueries[i++] = sq;}if (spanQueries.length == 1)return spanQueries[0];elsereturn new SpanOrQuery(spanQueries);}
public StashCreateCommand stashCreate() {return new StashCreateCommand(repo);}
public FieldInfo fieldInfo(String fieldName) {return byName.get(fieldName);}
public DescribeEventSourceResult describeEventSource(DescribeEventSourceRequest request) {request = beforeClientExecution(request);return executeDescribeEventSource(request);}
public GetDocumentAnalysisResult getDocumentAnalysis(GetDocumentAnalysisRequest request) {request = beforeClientExecution(request);return executeGetDocumentAnalysis(request);}
public CancelUpdateStackResult cancelUpdateStack(CancelUpdateStackRequest request) {request = beforeClientExecution(request);return executeCancelUpdateStack(request);}
public ModifyLoadBalancerAttributesResult modifyLoadBalancerAttributes(ModifyLoadBalancerAttributesRequest request) {request = beforeClientExecution(request);return executeModifyLoadBalancerAttributes(request);}
public SetInstanceProtectionResult setInstanceProtection(SetInstanceProtectionRequest request) {request = beforeClientExecution(request);return executeSetInstanceProtection(request);}
public ModifyDBProxyResult modifyDBProxy(ModifyDBProxyRequest request) {request = beforeClientExecution(request);return executeModifyDBProxy(request);}
public void add(char[] output, int offset, int len, int endOffset, int posLength) {if (count == outputs.length) {outputs = ArrayUtil.grow(outputs, count+1);}if (count == endOffsets.length) {final int[] next = new int[ArrayUtil.oversize(1+count, Integer.BYTES)];System.arraycopy(endOffsets, 0, next, 0, count);endOffsets = next;}if (count == posLengths.length) {final int[] next = new int[ArrayUtil.oversize(1+count, Integer.BYTES)];System.arraycopy(posLengths, 0, next, 0, count);posLengths = next;}if (outputs[count] == null) {outputs[count] = new CharsRefBuilder();}outputs[count].copyChars(output, offset, len);endOffsets[count] = endOffset;posLengths[count] = posLength;count++;}
public FetchLibrariesRequest() {super("CloudPhoto", "2017-07-11", "FetchLibraries", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public boolean exists() {return fs.exists(objects);}
public FilterOutputStream(OutputStream out) {this.out = out;}
public ScaleClusterRequest() {super("CS", "2015-12-15", "ScaleCluster", "csk");setUriPattern("/clusters/[ClusterId]");setMethod(MethodType.PUT);}
public DataValidationConstraint createTimeConstraint(int operatorType, String formula1, String formula2) {return DVConstraint.createTimeConstraint(operatorType, formula1, formula2);}
public ListObjectParentPathsResult listObjectParentPaths(ListObjectParentPathsRequest request) {request = beforeClientExecution(request);return executeListObjectParentPaths(request);}
public DescribeCacheSubnetGroupsResult describeCacheSubnetGroups(DescribeCacheSubnetGroupsRequest request) {request = beforeClientExecution(request);return executeDescribeCacheSubnetGroups(request);}
public void setSharedFormula(boolean flag) {field_5_options =sharedFormula.setShortBoolean(field_5_options, flag);}
public boolean isReuseObjects() {return reuseObjects;}
public ErrorNode addErrorNode(Token badToken) {ErrorNodeImpl t = new ErrorNodeImpl(badToken);addAnyChild(t);t.setParent(this);return t;}
public LatvianStemFilterFactory(Map<String,String> args) {super(args);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public EventSubscription removeSourceIdentifierFromSubscription(RemoveSourceIdentifierFromSubscriptionRequest request) {request = beforeClientExecution(request);return executeRemoveSourceIdentifierFromSubscription(request);}
public static TokenFilterFactory forName(String name, Map<String,String> args) {return loader.newInstance(name, args);}
public AddAlbumPhotosRequest() {super("CloudPhoto", "2017-07-11", "AddAlbumPhotos", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public GetThreatIntelSetResult getThreatIntelSet(GetThreatIntelSetRequest request) {request = beforeClientExecution(request);return executeGetThreatIntelSet(request);}
public RevFilter clone() {return new Binary(a.clone(), b.clone());}
public boolean equals( Object o ) {return o instanceof ArmenianStemmer;}
public final boolean hasArray() {return protectedHasArray();}
public UpdateContributorInsightsResult updateContributorInsights(UpdateContributorInsightsRequest request) {request = beforeClientExecution(request);return executeUpdateContributorInsights(request);}
public void unwriteProtectWorkbook() {records.remove(fileShare);records.remove(writeProtect);fileShare = null;writeProtect = null;}
public SolrSynonymParser(boolean dedup, boolean expand, Analyzer analyzer) {super(dedup, analyzer);this.expand = expand;}
public RequestSpotInstancesResult requestSpotInstances(RequestSpotInstancesRequest request) {request = beforeClientExecution(request);return executeRequestSpotInstances(request);}
public byte[] getObjectData() {return findObjectRecord().getObjectData();}
public GetContactAttributesResult getContactAttributes(GetContactAttributesRequest request) {request = beforeClientExecution(request);return executeGetContactAttributes(request);}
public String toString() {return getKey() + ": " + getValue(); }
public ListTextTranslationJobsResult listTextTranslationJobs(ListTextTranslationJobsRequest request) {request = beforeClientExecution(request);return executeListTextTranslationJobs(request);}
public GetContactMethodsResult getContactMethods(GetContactMethodsRequest request) {request = beforeClientExecution(request);return executeGetContactMethods(request);}
public static short lookupIndexByName(String name) {FunctionMetadata fd = getInstance().getFunctionByNameInternal(name);if (fd == null) {fd = getInstanceCetab().getFunctionByNameInternal(name);if (fd == null) {return -1;}}return (short) fd.getIndex();}
public DescribeAnomalyDetectorsResult describeAnomalyDetectors(DescribeAnomalyDetectorsRequest request) {request = beforeClientExecution(request);return executeDescribeAnomalyDetectors(request);}
public static String insertId(String message, ObjectId changeId) {return insertId(message, changeId, false);}
public long getObjectSize(AnyObjectId objectId, int typeHint)throws MissingObjectException, IncorrectObjectTypeException,IOException {long sz = db.getObjectSize(this, objectId);if (sz < 0) {if (typeHint == OBJ_ANY)throw new MissingObjectException(objectId.copy(),JGitText.get().unknownObjectType2);throw new MissingObjectException(objectId.copy(), typeHint);}return sz;}
public ImportInstallationMediaResult importInstallationMedia(ImportInstallationMediaRequest request) {request = beforeClientExecution(request);return executeImportInstallationMedia(request);}
public PutLifecycleEventHookExecutionStatusResult putLifecycleEventHookExecutionStatus(PutLifecycleEventHookExecutionStatusRequest request) {request = beforeClientExecution(request);return executePutLifecycleEventHookExecutionStatus(request);}
public NumberPtg(LittleEndianInput in)  {this(in.readDouble());}
public GetFieldLevelEncryptionConfigResult getFieldLevelEncryptionConfig(GetFieldLevelEncryptionConfigRequest request) {request = beforeClientExecution(request);return executeGetFieldLevelEncryptionConfig(request);}
public DescribeDetectorResult describeDetector(DescribeDetectorRequest request) {request = beforeClientExecution(request);return executeDescribeDetector(request);}
public ReportInstanceStatusResult reportInstanceStatus(ReportInstanceStatusRequest request) {request = beforeClientExecution(request);return executeReportInstanceStatus(request);}
public DeleteAlarmResult deleteAlarm(DeleteAlarmRequest request) {request = beforeClientExecution(request);return executeDeleteAlarm(request);}
public TokenStream create(TokenStream input) {return new PortugueseStemFilter(input);}
public FtCblsSubRecord() {reserved = new byte[ENCODED_SIZE];}
@Override public boolean remove(Object object) {synchronized (mutex) {return c.remove(object);}}
public GetDedicatedIpResult getDedicatedIp(GetDedicatedIpRequest request) {request = beforeClientExecution(request);return executeGetDedicatedIp(request);}
public String toString() {return precedence + " >= _p";}
public ListStreamProcessorsResult listStreamProcessors(ListStreamProcessorsRequest request) {request = beforeClientExecution(request);return executeListStreamProcessors(request);}
public DeleteLoadBalancerPolicyRequest(String loadBalancerName, String policyName) {setLoadBalancerName(loadBalancerName);setPolicyName(policyName);}
public WindowProtectRecord(int options) {_options = options;}
public UnbufferedCharStream(int bufferSize) {n = 0;data = new int[bufferSize];}
public GetOperationsResult getOperations(GetOperationsRequest request) {request = beforeClientExecution(request);return executeGetOperations(request);}
public void copyRawTo(byte[] b, int o) {NB.encodeInt32(b, o, w1);NB.encodeInt32(b, o + 4, w2);NB.encodeInt32(b, o + 8, w3);NB.encodeInt32(b, o + 12, w4);NB.encodeInt32(b, o + 16, w5);}
public WindowOneRecord(RecordInputStream in) {field_1_h_hold            = in.readShort();field_2_v_hold            = in.readShort();field_3_width             = in.readShort();field_4_height            = in.readShort();field_5_options           = in.readShort();field_6_active_sheet      = in.readShort();field_7_first_visible_tab = in.readShort();field_8_num_selected_tabs = in.readShort();field_9_tab_width_ratio   = in.readShort();}
public StopWorkspacesResult stopWorkspaces(StopWorkspacesRequest request) {request = beforeClientExecution(request);return executeStopWorkspaces(request);}
public void close() throws IOException {if (isOpen) {isOpen = false;try {dump();} finally {try {channel.truncate(fileLength);} finally {try {channel.close();} finally {fos.close();}}}}}
public DescribeMatchmakingRuleSetsResult describeMatchmakingRuleSets(DescribeMatchmakingRuleSetsRequest request) {request = beforeClientExecution(request);return executeDescribeMatchmakingRuleSets(request);}
public String getPronunciation(int wordId, char surface[], int off, int len) {return null; }
public String getPath() {return pathStr;}
public static double devsq(double[] v) {double r = Double.NaN;if (v!=null && v.length >= 1) {double m = 0;double s = 0;int n = v.length;for (int i=0; i<n; i++) {s += v[i];}m = s / n;s = 0;for (int i=0; i<n; i++) {s += (v[i]- m) * (v[i] - m);}r = (n == 1)? 0: s;}return r;}
public DescribeResizeResult describeResize(DescribeResizeRequest request) {request = beforeClientExecution(request);return executeDescribeResize(request);}
public final boolean hasPassedThroughNonGreedyDecision() {return passedThroughNonGreedyDecision;}
public int end() {return end(0);}
public void traverse(CellHandler handler) {int firstRow = range.getFirstRow();int lastRow = range.getLastRow();int firstColumn = range.getFirstColumn();int lastColumn = range.getLastColumn();final int width = lastColumn - firstColumn + 1;SimpleCellWalkContext ctx = new SimpleCellWalkContext();Row currentRow = null;Cell currentCell = null;for (ctx.rowNumber = firstRow; ctx.rowNumber <= lastRow; ++ctx.rowNumber) {currentRow = sheet.getRow(ctx.rowNumber);if (currentRow == null) {continue;}for (ctx.colNumber = firstColumn; ctx.colNumber <= lastColumn; ++ctx.colNumber) {currentCell = currentRow.getCell(ctx.colNumber);if (currentCell == null) {continue;}if (isEmpty(currentCell) && !traverseEmptyCells) {continue;}long rowSize = ArithmeticUtils.mulAndCheck((long)ArithmeticUtils.subAndCheck(ctx.rowNumber, firstRow), (long)width);ctx.ordinalNumber = ArithmeticUtils.addAndCheck(rowSize, (ctx.colNumber - firstColumn + 1));handler.onCell(currentCell, ctx);}}}
public int getReadIndex() {return pos;}
public int compareTo(ScoreTerm other) {if (this.boost == other.boost)return other.bytes.get().compareTo(this.bytes.get());elsereturn Float.compare(this.boost, other.boost);}
public int normalize(char s[], int len) {for (int i = 0; i < len; i++) {switch (s[i]) {case FARSI_YEH:case YEH_BARREE:s[i] = YEH;break;case KEHEH:s[i] = KAF;break;case HEH_YEH:case HEH_GOAL:s[i] = HEH;break;case HAMZA_ABOVE: len = delete(s, i, len);i--;break;default:break;}}return len;}
public void serialize(LittleEndianOutput out) {out.writeShort(_options);}
public DiagnosticErrorListener(boolean exactOnly) {this.exactOnly = exactOnly;}
public KeySchemaElement(String attributeName, KeyType keyType) {setAttributeName(attributeName);setKeyType(keyType.toString());}
public GetAssignmentResult getAssignment(GetAssignmentRequest request) {request = beforeClientExecution(request);return executeGetAssignment(request);}
public boolean hasObject(AnyObjectId id) {return findOffset(id) != -1;}
public GroupingSearch setAllGroups(boolean allGroups) {this.allGroups = allGroups;return this;}
public synchronized void setMultiValued(String dimName, boolean v) {DimConfig ft = fieldTypes.get(dimName);if (ft == null) {ft = new DimConfig();fieldTypes.put(dimName, ft);}ft.multiValued = v;}
public int getCellsVal() {Iterator<Character> i = cells.keySet().iterator();int size = 0;for (; i.hasNext();) {Character c = i.next();Cell e = at(c);if (e.cmd >= 0) {size++;}}return size;}
public DeleteVoiceConnectorResult deleteVoiceConnector(DeleteVoiceConnectorRequest request) {request = beforeClientExecution(request);return executeDeleteVoiceConnector(request);}
public DeleteLifecyclePolicyResult deleteLifecyclePolicy(DeleteLifecyclePolicyRequest request) {request = beforeClientExecution(request);return executeDeleteLifecyclePolicy(request);}
public void write(byte[] b) {int len = b.length;checkPosition(len);System.arraycopy(b, 0, _buf, _writeIndex, len);_writeIndex += len;}
public RebaseResult getRebaseResult() {return this.rebaseResult;}
public static int getNearestSetSize(int maxNumberOfValuesExpected,float desiredSaturation) {for (int i = 0; i < usableBitSetSizes.length; i++) {int numSetBitsAtDesiredSaturation = (int) (usableBitSetSizes[i] * desiredSaturation);int estimatedNumUniqueValues = getEstimatedNumberUniqueValuesAllowingForCollisions(usableBitSetSizes[i], numSetBitsAtDesiredSaturation);if (estimatedNumUniqueValues > maxNumberOfValuesExpected) {return usableBitSetSizes[i];}}return -1;}
public DescribeDashboardResult describeDashboard(DescribeDashboardRequest request) {request = beforeClientExecution(request);return executeDescribeDashboard(request);}
public CreateSegmentResult createSegment(CreateSegmentRequest request) {request = beforeClientExecution(request);return executeCreateSegment(request);}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[DBCELL]\n");buffer.append("    .rowoffset = ").append(HexDump.intToHex(field_1_row_offset)).append("\n");for (int k = 0; k < field_2_cell_offsets.length; k++) {buffer.append("    .cell_").append(k).append(" = ").append(HexDump.shortToHex(field_2_cell_offsets[ k ])).append("\n");}buffer.append("[/DBCELL]\n");return buffer.toString();}
public List<String> getUndeletedList() {return undeletedList;}
public String toString() {return "[INTERFACEEND/]\n";}
public MergeScheduler clone() {return this;}
public PlainTextDictionary(Reader reader) {in = new BufferedReader(reader);}
public StringBuilder append(CharSequence csq) {if (csq == null) {appendNull();} else {append0(csq, 0, csq.length());}return this;}
public ListAssociatedStacksResult listAssociatedStacks(ListAssociatedStacksRequest request) {request = beforeClientExecution(request);return executeListAssociatedStacks(request);}
public static double avedev(double[] v) {double r = 0;double m = 0;double s = 0;for (int i=0, iSize=v.length; i<iSize; i++) {s += v[i];}m = s / v.length;s = 0;for (int i=0, iSize=v.length; i<iSize; i++) {s += Math.abs(v[i]-m);}r = s / v.length;return r;}
public DescribeByoipCidrsResult describeByoipCidrs(DescribeByoipCidrsRequest request) {request = beforeClientExecution(request);return executeDescribeByoipCidrs(request);}
public GetDiskResult getDisk(GetDiskRequest request) {request = beforeClientExecution(request);return executeGetDisk(request);}
public DBClusterParameterGroup createDBClusterParameterGroup(CreateDBClusterParameterGroupRequest request) {request = beforeClientExecution(request);return executeCreateDBClusterParameterGroup(request);}
public static CharBuffer wrap(char[] array, int start, int charCount) {Arrays.checkOffsetAndCount(array.length, start, charCount);CharBuffer buf = new ReadWriteCharArrayBuffer(array);buf.position = start;buf.limit = start + charCount;return buf;}
public SubmoduleStatusType getType() {return type;}
public DescribeGameServerGroupResult describeGameServerGroup(DescribeGameServerGroupRequest request) {request = beforeClientExecution(request);return executeDescribeGameServerGroup(request);}
public Pattern pattern() {return pattern;}
public V setValue(V object) {throw new UnsupportedOperationException();}
public StringBuilder stem(CharSequence word) {CharSequence cmd = stemmer.getLastOnPath(word);if (cmd == null)return null;buffer.setLength(0);buffer.append(word);Diff.apply(buffer, cmd);if (buffer.length() > 0)return buffer;elsereturn null;}
public RenameFaceRequest() {super("CloudPhoto", "2017-07-11", "RenameFace", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public char requireChar(Map<String,String> args, String name) {return require(args, name).charAt(0);}
public static String toStringTree(Tree t) {return toStringTree(t, (List<String>)null);}
public String toString() {return "<deleted/>";}
public GetRepoWebhookLogListRequest() {super("cr", "2016-06-07", "GetRepoWebhookLogList", "cr");setUriPattern("/repos/[RepoNamespace]/[RepoName]/webhooks/[WebhookId]/logs");setMethod(MethodType.GET);}
public GetJobUnlockCodeResult getJobUnlockCode(GetJobUnlockCodeRequest request) {request = beforeClientExecution(request);return executeGetJobUnlockCode(request);}
public RemoveTagsRequest(String resourceId) {setResourceId(resourceId);}
public short getGB2312Id(char ch) {try {byte[] buffer = Character.toString(ch).getBytes("GB2312");if (buffer.length != 2) {return -1;}int b0 = (buffer[0] & 0x0FF) - 161; int b1 = (buffer[1] & 0x0FF) - 161; return (short) (b0 * 94 + b1);} catch (UnsupportedEncodingException e) {throw new RuntimeException(e);}}
public BatchRefUpdate addCommand(Collection<ReceiveCommand> cmd) {commands.addAll(cmd);return this;}
public short checkExternSheet(int sheetNumber){return (short)getOrCreateLinkTable().checkExternSheet(sheetNumber);}
@Override public boolean equals(Object object) {return c.equals(object);}
public BooleanQuery build(QueryNode queryNode) throws QueryNodeException {AnyQueryNode andNode = (AnyQueryNode) queryNode;BooleanQuery.Builder bQuery = new BooleanQuery.Builder();List<QueryNode> children = andNode.getChildren();if (children != null) {for (QueryNode child : children) {Object obj = child.getTag(QueryTreeBuilder.QUERY_TREE_BUILDER_TAGID);if (obj != null) {Query query = (Query) obj;try {bQuery.add(query, BooleanClause.Occur.SHOULD);} catch (TooManyClauses ex) {throw new QueryNodeException(new MessageImpl(QueryParserMessages.EMPTY_MESSAGE), ex);}}}}bQuery.setMinimumNumberShouldMatch(andNode.getMinimumMatchingElements());return bQuery.build();}
public DescribeStreamProcessorResult describeStreamProcessor(DescribeStreamProcessorRequest request) {request = beforeClientExecution(request);return executeDescribeStreamProcessor(request);}
public DescribeDashboardPermissionsResult describeDashboardPermissions(DescribeDashboardPermissionsRequest request) {request = beforeClientExecution(request);return executeDescribeDashboardPermissions(request);}
public Ref peel(Ref ref) {try {return getRefDatabase().peel(ref);} catch (IOException e) {return ref;}}
public long ramBytesUsed() {return RamUsageEstimator.alignObjectSize(RamUsageEstimator.NUM_BYTES_OBJECT_HEADER+ 2 * Integer.BYTES     + RamUsageEstimator.NUM_BYTES_OBJECT_REF) + RamUsageEstimator.sizeOf(blocks);}
public GetDomainSuggestionsResult getDomainSuggestions(GetDomainSuggestionsRequest request) {request = beforeClientExecution(request);return executeGetDomainSuggestions(request);}
public DescribeStackEventsResult describeStackEvents(DescribeStackEventsRequest request) {request = beforeClientExecution(request);return executeDescribeStackEvents(request);}
public void setRule(int idx, ConditionalFormattingRule cfRule){setRule(idx, (HSSFConditionalFormattingRule)cfRule);}
public CreateResolverRuleResult createResolverRule(CreateResolverRuleRequest request) {request = beforeClientExecution(request);return executeCreateResolverRule(request);}
public SeriesIndexRecord(RecordInputStream in) {field_1_index = in.readShort();}
public GetStylesRequest() {super("lubancloud", "2018-05-09", "GetStyles", "luban");setMethod(MethodType.POST);}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_gridset_flag);}
public boolean equals(Object obj) {if (this == obj) {return true;}if (obj == null) {return false;}if (getClass() != obj.getClass()) {return false;}Toffs other = (Toffs) obj;if (getStartOffset() != other.getStartOffset()) {return false;}if (getEndOffset() != other.getEndOffset()) {return false;}return true;}
public CreateGatewayGroupResult createGatewayGroup(CreateGatewayGroupRequest request) {request = beforeClientExecution(request);return executeCreateGatewayGroup(request);}
public CreateParticipantConnectionResult createParticipantConnection(CreateParticipantConnectionRequest request) {request = beforeClientExecution(request);return executeCreateParticipantConnection(request);}
public static double irr(double[] income) {return irr(income, 0.1d);}
public RegisterWorkspaceDirectoryResult registerWorkspaceDirectory(RegisterWorkspaceDirectoryRequest request) {request = beforeClientExecution(request);return executeRegisterWorkspaceDirectory(request);}
public RevertCommand include(AnyObjectId commit) {return include(commit.getName(), commit);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval inumberVE) {ValueEval veText1;try {veText1 = OperandResolver.getSingleValue(inumberVE, srcRowIndex, srcColumnIndex);} catch (EvaluationException e) {return e.getErrorEval();}String iNumber = OperandResolver.coerceValueToString(veText1);Matcher m = COMPLEX_NUMBER_PATTERN.matcher(iNumber);boolean result = m.matches();String imaginary = "";if (result) {String imaginaryGroup = m.group(5);boolean hasImaginaryPart = imaginaryGroup.equals("i") || imaginaryGroup.equals("j");if (imaginaryGroup.length() == 0) {return new StringEval(String.valueOf(0));}if (hasImaginaryPart) {String sign = "";String imaginarySign = m.group(GROUP3_IMAGINARY_SIGN);if (imaginarySign.length() != 0 && !(imaginarySign.equals("+"))) {sign = imaginarySign;}String groupImaginaryNumber = m.group(GROUP4_IMAGINARY_INTEGER_OR_DOUBLE);if (groupImaginaryNumber.length() != 0) {imaginary = sign + groupImaginaryNumber;} else {imaginary = sign + "1";}}} else {return ErrorEval.NUM_ERROR;}return new StringEval(imaginary);}
public E pollLast() {Map.Entry<E, Object> entry = backingMap.pollLastEntry();return (entry == null) ? null : entry.getKey();}
public int readUShort(){int ch1 = readUByte();int ch2 = readUByte();return (ch2 << 8) + (ch1 << 0);}
public ModifySnapshotAttributeRequest(String snapshotId, SnapshotAttributeName attribute, OperationType operationType) {setSnapshotId(snapshotId);setAttribute(attribute.toString());setOperationType(operationType.toString());}
public ListBonusPaymentsResult listBonusPayments(ListBonusPaymentsRequest request) {request = beforeClientExecution(request);return executeListBonusPayments(request);}
public V get(CharSequence cs) {if(cs == null)throw new NullPointerException();return null;}
public TokenFilter create(TokenStream input) {CommonGramsFilter commonGrams = (CommonGramsFilter) super.create(input);return new CommonGramsQueryFilter(commonGrams);}
public String getPath() {return path;}
public InitiateMultipartUploadResult initiateMultipartUpload(InitiateMultipartUploadRequest request) {request = beforeClientExecution(request);return executeInitiateMultipartUpload(request);}
public StringBuilder insert(int offset, int i) {insert0(offset, Integer.toString(i));return this;}
public void decode(long[] blocks, int blocksOffset, int[] values, int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final long block = blocks[blocksOffset++];for (int shift = 62; shift >= 0; shift -= 2) {values[valuesOffset++] = (int) ((block >>> shift) & 3);}}}
public TokenStream create(TokenStream input) {return new ElisionFilter(input, articles);}
public boolean eat(Row in, int remap[]) {int sum = 0;for (Iterator<Cell> i = in.cells.values().iterator(); i.hasNext();) {Cell c = i.next();sum += c.cnt;if (c.ref >= 0) {if (remap[c.ref] == 0) {c.ref = -1;}}}int frame = sum / 10;boolean live = false;for (Iterator<Cell> i = in.cells.values().iterator(); i.hasNext();) {Cell c = i.next();if (c.cnt < frame && c.cmd >= 0) {c.cnt = 0;c.cmd = -1;}if (c.cmd >= 0 || c.ref >= 0) {live |= true;}}return !live;}
final public Token getToken(int index) {Token t = jj_lookingAhead ? jj_scanpos : token;for (int i = 0; i < index; i++) {if (t.next != null) t = t.next;else t = t.next = token_source.getNextToken();}return t;}
public String toString() {StringBuilder sb = new StringBuilder();sb.append(getClass().getName()).append(" [ARRAY]\n");sb.append(" range=").append(getRange()).append("\n");sb.append(" options=").append(HexDump.shortToHex(_options)).append("\n");sb.append(" notUsed=").append(HexDump.intToHex(_field3notUsed)).append("\n");sb.append(" formula:").append("\n");Ptg[] ptgs = _formula.getTokens();for (int i = 0; i < ptgs.length; i++) {Ptg ptg = ptgs[i];sb.append(ptg).append(ptg.getRVAType()).append("\n");}sb.append("]");return sb.toString();}
public GetFolderResult getFolder(GetFolderRequest request) {request = beforeClientExecution(request);return executeGetFolder(request);}
@Override public void add(int location, E object) {throw new UnsupportedOperationException();}
public PositiveScoresOnlyCollector(Collector in) {super(in);}
public CreateRepoBuildRuleRequest() {super("cr", "2016-06-07", "CreateRepoBuildRule", "cr");setUriPattern("/repos/[RepoNamespace]/[RepoName]/rules");setMethod(MethodType.PUT);}
public BaseRef(AreaEval ae) {_refEval = null;_areaEval = ae;_firstRowIndex = ae.getFirstRow();_firstColumnIndex = ae.getFirstColumn();_height = ae.getLastRow() - ae.getFirstRow() + 1;_width = ae.getLastColumn() - ae.getFirstColumn() + 1;}
public DrawingManager2( EscherDggRecord dgg ) {this.dgg = dgg;}
public void reset() {if (!first())reset(raw);}
public final CharsetDecoder reset() {status = INIT;implReset();return this;}
public BufferedReader(Reader in, int size) {super(in);if (size <= 0) {throw new IllegalArgumentException("size <= 0");}this.in = in;buf = new char[size];}
public DescribeCodeRepositoryResult describeCodeRepository(DescribeCodeRepositoryRequest request) {request = beforeClientExecution(request);return executeDescribeCodeRepository(request);}
public DBSubnetGroup createDBSubnetGroup(CreateDBSubnetGroupRequest request) {request = beforeClientExecution(request);return executeCreateDBSubnetGroup(request);}
public RenameBranchCommand setOldName(String oldName) {checkCallable();this.oldName = oldName;return this;}
public DeleteBranchCommand setForce(boolean force) {checkCallable();this.force = force;return this;}
public StopCompilationJobResult stopCompilationJob(StopCompilationJobRequest request) {request = beforeClientExecution(request);return executeStopCompilationJob(request);}
public synchronized final void incrementSecondaryProgressBy(int diff) {setSecondaryProgress(mSecondaryProgress + diff);}
public int[] clear() {return bytesStart = null;}
public String getRawPath() {return path;}
public GetUserSourceAccountRequest() {super("cr", "2016-06-07", "GetUserSourceAccount", "cr");setUriPattern("/users/sourceAccount");setMethod(MethodType.GET);}
public CreateExportJobResult createExportJob(CreateExportJobRequest request) {request = beforeClientExecution(request);return executeCreateExportJob(request);}
public CreateDedicatedIpPoolResult createDedicatedIpPool(CreateDedicatedIpPoolRequest request) {request = beforeClientExecution(request);return executeCreateDedicatedIpPool(request);}
public boolean equals(Object obj) {if (this == obj) {return true;}if (obj == null) {return false;}if (obj instanceof HSSFCellStyle) {final HSSFCellStyle other = (HSSFCellStyle) obj;if (_format == null) {if (other._format != null) {return false;}} else if (!_format.equals(other._format)) {return false;}if (_index != other._index) {return false;}return true;}return false;}
public ReleaseHostsResult releaseHosts(ReleaseHostsRequest request) {request = beforeClientExecution(request);return executeReleaseHosts(request);}
public boolean equals(Object object) {if (this == object) {return true;}if (object instanceof Set) {Set<?> s = (Set<?>) object;try {return size() == s.size() && containsAll(s);} catch (NullPointerException ignored) {return false;} catch (ClassCastException ignored) {return false;}}return false;}
public void setRefLogMessage(String msg, boolean appendStatus) {customRefLog = true;if (msg == null && !appendStatus) {disableRefLog();} else if (msg == null && appendStatus) {refLogMessage = ""; refLogIncludeResult = true;} else {refLogMessage = msg;refLogIncludeResult = appendStatus;}}
public StreamIDRecord(RecordInputStream in) {idstm = in.readShort();}
public RecognizeCarRequest() {super("visionai-poc", "2020-04-08", "RecognizeCar");setMethod(MethodType.POST);}
public final ByteOrder order() {return ByteOrder.nativeOrder();}
public int getAheadCount() {return aheadCount;}
public boolean isNewFragment() {return false;}
public GetCloudFrontOriginAccessIdentityConfigResult getCloudFrontOriginAccessIdentityConfig(GetCloudFrontOriginAccessIdentityConfigRequest request) {request = beforeClientExecution(request);return executeGetCloudFrontOriginAccessIdentityConfig(request);}
public boolean matches(int symbol, int minVocabSymbol, int maxVocabSymbol) {return label == symbol;}
public DeleteTransitGatewayResult deleteTransitGateway(DeleteTransitGatewayRequest request) {request = beforeClientExecution(request);return executeDeleteTransitGateway(request);}
public static byte[] grow(byte[] array, int minSize) {assert minSize >= 0: "size must be positive (got " + minSize + "): likely integer overflow?";if (array.length < minSize) {return growExact(array, oversize(minSize, Byte.BYTES));} elsereturn array;}
public CreateTransactionRequest() {super("CloudPhoto", "2017-07-11", "CreateTransaction", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public BatchRefUpdate setRefLogIdent(PersonIdent pi) {refLogIdent = pi;return this;}
public GetLaunchTemplateDataResult getLaunchTemplateData(GetLaunchTemplateDataRequest request) {request = beforeClientExecution(request);return executeGetLaunchTemplateData(request);}
public ParseInfo(ProfilingATNSimulator atnSimulator) {this.atnSimulator = atnSimulator;}
public SimpleQQParser(String qqNames[], String indexField) {this.qqNames = qqNames;this.indexField = indexField;}
public DBCluster promoteReadReplicaDBCluster(PromoteReadReplicaDBClusterRequest request) {request = beforeClientExecution(request);return executePromoteReadReplicaDBCluster(request);}
public DescribeCapacityReservationsResult describeCapacityReservations(DescribeCapacityReservationsRequest request) {request = beforeClientExecution(request);return executeDescribeCapacityReservations(request);}
public String toString() {return "IndexSearcher(" + reader + "; executor=" + executor + "; sliceExecutionControlPlane " + sliceExecutor + ")";}
public final boolean incrementToken() {return false;}
public void serialize(LittleEndianOutput out) {out.writeShort(main + 1);out.writeShort(subFrom);out.writeShort(subTo);}
public void decode(byte[] blocks, int blocksOffset, int[] values,int valuesOffset, int iterations) {if (bitsPerValue > 32) {throw new UnsupportedOperationException("Cannot decode " + bitsPerValue + "-bits values into an int[]");}for (int i = 0; i < iterations; ++i) {final long block = readLong(blocks, blocksOffset);blocksOffset += 8;valuesOffset = decode(block, values, valuesOffset);}}
public boolean isExpectedToken(int symbol) {ATN atn = getInterpreter().atn;ParserRuleContext ctx = _ctx;ATNState s = atn.states.get(getState());IntervalSet following = atn.nextTokens(s);if (following.contains(symbol)) {return true;}if ( !following.contains(Token.EPSILON) ) return false;while ( ctx!=null && ctx.invokingState>=0 && following.contains(Token.EPSILON) ) {ATNState invokingState = atn.states.get(ctx.invokingState);RuleTransition rt = (RuleTransition)invokingState.transition(0);following = atn.nextTokens(rt.followState);if (following.contains(symbol)) {return true;}ctx = (ParserRuleContext)ctx.parent;}if ( following.contains(Token.EPSILON) && symbol == Token.EOF ) {return true;}return false;}
public UpdateStreamResult updateStream(UpdateStreamRequest request) {request = beforeClientExecution(request);return executeUpdateStream(request);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0) {try {OperandResolver.getSingleValue(arg0, srcRowIndex, srcColumnIndex);return ErrorEval.NA;} catch (EvaluationException e) {int result = translateErrorCodeToErrorTypeValue(e.getErrorEval().getErrorCode());return new NumberEval(result);}}
public String toString() {return getClass().getName() + " [" + _index + " " + _name + "]";}
public ListAssignmentsForHITResult listAssignmentsForHIT(ListAssignmentsForHITRequest request) {request = beforeClientExecution(request);return executeListAssignmentsForHIT(request);}
public DeleteAccessControlRuleResult deleteAccessControlRule(DeleteAccessControlRuleRequest request) {request = beforeClientExecution(request);return executeDeleteAccessControlRule(request);}
public Arc<Long> getFirstArc(FST.Arc<Long> arc) {return fst.getFirstArc(arc);}
public void decode(long[] blocks, int blocksOffset, int[] values, int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final long block = blocks[blocksOffset++];for (int shift = 48; shift >= 0; shift -= 16) {values[valuesOffset++] = (int) ((block >>> shift) & 65535);}}}
public long skip(long charCount) throws IOException {if (charCount < 0) {throw new IllegalArgumentException("charCount < 0: " + charCount);}synchronized (lock) {checkNotClosed();if (charCount == 0) {return 0;}long inSkipped;int availableFromBuffer = buf.length - pos;if (availableFromBuffer > 0) {long requiredFromIn = charCount - availableFromBuffer;if (requiredFromIn <= 0) {pos += charCount;return charCount;}pos += availableFromBuffer;inSkipped = in.skip(requiredFromIn);} else {inSkipped = in.skip(charCount);}return inSkipped + availableFromBuffer;}}
public Map<String, Ref> getRefsMap() {return advertisedRefs;}
public UpdateApiKeyResult updateApiKey(UpdateApiKeyRequest request) {request = beforeClientExecution(request);return executeUpdateApiKey(request);}
public ObjectStream openStream() throws MissingObjectException, IOException {PackInputStream packIn;@SuppressWarnings("resource")DfsReader ctx = db.newReader();try {try {packIn = new PackInputStream(pack, objectOffset + headerLength, ctx);ctx = null; } catch (IOException packGone) {ObjectId obj = pack.getReverseIdx(ctx).findObject(objectOffset);return ctx.open(obj, type).openStream();}} finally {if (ctx != null) {ctx.close();}}int bufsz = 8192;InputStream in = new BufferedInputStream(new InflaterInputStream(packIn, packIn.ctx.inflater(), bufsz),bufsz);return new ObjectStream.Filter(type, size, in);}
public ArrayList() {array = EmptyArray.OBJECT;}
public UpdateDetectorVersionResult updateDetectorVersion(UpdateDetectorVersionRequest request) {request = beforeClientExecution(request);return executeUpdateDetectorVersion(request);}
public void resize(){resize(Double.MAX_VALUE);}
public RevFlagSet(Collection<RevFlag> s) {this();addAll(s);}
public int size() {return size;}
public final long getLong() {int newPosition = position + SizeOf.LONG;if (newPosition > limit) {throw new BufferUnderflowException();}long result = Memory.peekLong(backingArray, offset + position, order);position = newPosition;return result;}
public StringBuilder insert(int offset, long l) {insert0(offset, Long.toString(l));return this;}
public TurkishLowerCaseFilter(TokenStream in) {super(in);}
public ParseTreeMatch match(ParseTree tree, ParseTreePattern pattern) {MultiMap<String, ParseTree> labels = new MultiMap<String, ParseTree>();ParseTree mismatchedNode = matchImpl(tree, pattern.getPatternTree(), labels);return new ParseTreeMatch(tree, pattern, labels, mismatchedNode);}
public void addIfNoOverlap( WeightedPhraseInfo wpi ){for( WeightedPhraseInfo existWpi : getPhraseList() ){if( existWpi.isOffsetOverlap( wpi ) ) {existWpi.getTermsInfos().addAll( wpi.getTermsInfos() );return;}}getPhraseList().add( wpi );}
public ThreeWayMerger newMerger(Repository db) {return new InCoreMerger(db);}
public float docScore(int docId, String field, int numPayloadsSeen, float payloadScore) {return numPayloadsSeen > 0 ? (payloadScore / numPayloadsSeen) : 1;}
public Collection<ParseTree> evaluate(ParseTree t) {return Trees.findAllRuleNodes(t, ruleIndex);}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[CFRULE]\n");buffer.append("    .condition_type   =").append(getConditionType()).append("\n");buffer.append("    OPTION FLAGS=0x").append(Integer.toHexString(getOptions())).append("\n");if (containsFontFormattingBlock()) {buffer.append(_fontFormatting).append("\n");}if (containsBorderFormattingBlock()) {buffer.append(_borderFormatting).append("\n");}if (containsPatternFormattingBlock()) {buffer.append(_patternFormatting).append("\n");}buffer.append("    Formula 1 =").append(Arrays.toString(getFormula1().getTokens())).append("\n");buffer.append("    Formula 2 =").append(Arrays.toString(getFormula2().getTokens())).append("\n");buffer.append("[/CFRULE]\n");return buffer.toString();}
public DescribeServiceUpdatesResult describeServiceUpdates(DescribeServiceUpdatesRequest request) {request = beforeClientExecution(request);return executeDescribeServiceUpdates(request);}
public String getNameName(int index){return getNameAt(index).getNameName();}
public DescribeLocationsResult describeLocations() {return describeLocations(new DescribeLocationsRequest());}
public String toString() {return "<phraseslop value='" + getValueString() + "'>" + "\n"+ getChild().toString() + "\n</phraseslop>";}
public DirCacheEntry getDirCacheEntry() {return currentSubtree == null ? currentEntry : null;}
public IntBuffer put(int[] src, int srcOffset, int intCount) {Arrays.checkOffsetAndCount(src.length, srcOffset, intCount);if (intCount > remaining()) {throw new BufferOverflowException();}for (int i = srcOffset; i < srcOffset + intCount; ++i) {put(src[i]);}return this;}
public void trimToSize() {int s = size;if (s == array.length) {return;}if (s == 0) {array = EmptyArray.OBJECT;} else {Object[] newArray = new Object[s];System.arraycopy(array, 0, newArray, 0, s);array = newArray;}modCount++;}
public DescribeLocalGatewayVirtualInterfacesResult describeLocalGatewayVirtualInterfaces(DescribeLocalGatewayVirtualInterfacesRequest request) {request = beforeClientExecution(request);return executeDescribeLocalGatewayVirtualInterfaces(request);}
public TokenStream create(TokenStream input) {return new RussianLightStemFilter(input);}
public int [] toArray(final int [] a){int[] rval;if (a.length == _limit){System.arraycopy(_array, 0, a, 0, _limit);rval = a;}else{rval = toArray();}return rval;}
public BasicSessionCredentials(String accessKeyId, String accessKeySecret, String sessionToken,long roleSessionDurationSeconds) {if (accessKeyId == null) {throw new IllegalArgumentException("Access key ID cannot be null.");}if (accessKeySecret == null) {throw new IllegalArgumentException("Access key secret cannot be null.");}this.accessKeyId = accessKeyId;this.accessKeySecret = accessKeySecret;this.sessionToken = sessionToken;this.roleSessionDurationSeconds = roleSessionDurationSeconds;this.sessionStartedTimeInMilliSeconds = System.currentTimeMillis();}
public final ShortBuffer get(short[] dst, int dstOffset, int shortCount) {if (shortCount > remaining()) {throw new BufferUnderflowException();}System.arraycopy(backingArray, offset + position, dst, dstOffset, shortCount);position += shortCount;return this;}
public ActivateEventSourceResult activateEventSource(ActivateEventSourceRequest request) {request = beforeClientExecution(request);return executeActivateEventSource(request);}
public DescribeReceiptRuleSetResult describeReceiptRuleSet(DescribeReceiptRuleSetRequest request) {request = beforeClientExecution(request);return executeDescribeReceiptRuleSet(request);}
public Filter(String name) {setName(name);}
public DoubleBuffer put(double c) {throw new ReadOnlyBufferException();}
public CreateTrafficPolicyInstanceResult createTrafficPolicyInstance(CreateTrafficPolicyInstanceRequest request) {request = beforeClientExecution(request);return executeCreateTrafficPolicyInstance(request);}
public JapaneseIterationMarkCharFilter(Reader input, boolean normalizeKanji, boolean normalizeKana) {super(input);this.normalizeKanji = normalizeKanji;this.normalizeKana = normalizeKana;buffer.reset(input);}
public void writeLong(long v) {writeInt((int)(v >>  0));writeInt((int)(v >> 32));}
public FileResolver() {exports = new ConcurrentHashMap<>();exportBase = new CopyOnWriteArrayList<>();}
public ValueEval getRef3DEval(Ref3DPxg rptg) {SheetRangeEvaluator sre = createExternSheetRefEvaluator(rptg.getSheetName(), rptg.getLastSheetName(), rptg.getExternalWorkbookNumber());return new LazyRefEval(rptg.getRow(), rptg.getColumn(), sre);}
public DeleteDatasetResult deleteDataset(DeleteDatasetRequest request) {request = beforeClientExecution(request);return executeDeleteDataset(request);}
public StartRelationalDatabaseResult startRelationalDatabase(StartRelationalDatabaseRequest request) {request = beforeClientExecution(request);return executeStartRelationalDatabase(request);}
public DescribeReservedCacheNodesOfferingsResult describeReservedCacheNodesOfferings() {return describeReservedCacheNodesOfferings(new DescribeReservedCacheNodesOfferingsRequest());}
static public double pmt(double r, int nper, double pv, double fv, int type) {return -r * (pv * Math.pow(1 + r, nper) + fv) / ((1 + r*type) * (Math.pow(1 + r, nper) - 1));}
public DescribeDocumentVersionsResult describeDocumentVersions(DescribeDocumentVersionsRequest request) {request = beforeClientExecution(request);return executeDescribeDocumentVersions(request);}
public ListPublishingDestinationsResult listPublishingDestinations(ListPublishingDestinationsRequest request) {request = beforeClientExecution(request);return executeListPublishingDestinations(request);}
public DeleteAccountAliasRequest(String accountAlias) {setAccountAlias(accountAlias);}
public static long[] grow(long[] array) {return grow(array, 1 + array.length);}
public String outputToString(Object output) {if (!(output instanceof List)) {return outputs.outputToString((T) output);} else {List<T> outputList = (List<T>) output;StringBuilder b = new StringBuilder();b.append('[');for(int i=0;i<outputList.size();i++) {if (i > 0) {b.append(", ");}b.append(outputs.outputToString(outputList.get(i)));}b.append(']');return b.toString();}}
public void notifyDeleteCell(Cell cell) {_bookEvaluator.notifyDeleteCell(new HSSFEvaluationCell((HSSFCell)cell));}
public StringBuilder replace(int start, int end, String str) {replace0(start, end, str);return this;}
public SetIdentityPoolConfigurationResult setIdentityPoolConfiguration(SetIdentityPoolConfigurationRequest request) {request = beforeClientExecution(request);return executeSetIdentityPoolConfiguration(request);}
public static double kthSmallest(double[] v, int k) {double r = Double.NaN;int index = k-1; if (v!=null && v.length > index && index >= 0) {Arrays.sort(v);r = v[index];}return r;}
public void set(int index, long value) {final int o = index >>> 5;final int b = index & 31;final int shift = b << 1;blocks[o] = (blocks[o] & ~(3L << shift)) | (value << shift);}
public String toString() {if (getChildren() == null || getChildren().size() == 0)return "<boolean operation='and'/>";StringBuilder sb = new StringBuilder();sb.append("<boolean operation='and'>");for (QueryNode child : getChildren()) {sb.append("\n");sb.append(child.toString());}sb.append("\n</boolean>");return sb.toString();}
public int sumTokenSizes(int fromIx, int toIx) {int result = 0;for (int i=fromIx; i<toIx; i++) {result += _ptgs[i].getSize();}return result;}
public void setReadonly(boolean readonly) {if ( this.readonly && !readonly ) throw new IllegalStateException("can't alter readonly IntervalSet");this.readonly = readonly;}
public final void clearConsumingCell(FormulaCellCacheEntry cce) {if(!_consumingCells.remove(cce)) {throw new IllegalStateException("Specified formula cell is not consumed by this cell");}}
@Override public List<E> subList(int start, int end) {synchronized (mutex) {return new SynchronizedRandomAccessList<E>(list.subList(start, end), mutex);}}
public FileHeader getFileHeader() {return file;}
public AttachLoadBalancersResult attachLoadBalancers(AttachLoadBalancersRequest request) {request = beforeClientExecution(request);return executeAttachLoadBalancers(request);}
public InitiateJobRequest(String accountId, String vaultName, JobParameters jobParameters) {setAccountId(accountId);setVaultName(vaultName);setJobParameters(jobParameters);}
public String toString() {return "SPL";}
public ReplaceableAttribute(String name, String value, Boolean replace) {setName(name);setValue(value);setReplace(replace);}
public final void add(IndexableField field) {fields.add(field);}
public DeleteStackSetResult deleteStackSet(DeleteStackSetRequest request) {request = beforeClientExecution(request);return executeDeleteStackSet(request);}
public GetRepoBuildRuleListRequest() {super("cr", "2016-06-07", "GetRepoBuildRuleList", "cr");setUriPattern("/repos/[RepoNamespace]/[RepoName]/rules");setMethod(MethodType.GET);}
public SparseArray(int initialCapacity) {initialCapacity = ArrayUtils.idealIntArraySize(initialCapacity);mKeys = new int[initialCapacity];mValues = new Object[initialCapacity];mSize = 0;}
public InvokeServiceRequest() {super("industry-brain", "2018-07-12", "InvokeService");setMethod(MethodType.POST);}
public ListAlbumPhotosRequest() {super("CloudPhoto", "2017-07-11", "ListAlbumPhotos", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public boolean hasPrevious() {return link != list.voidLink;}
public DeleteHsmConfigurationResult deleteHsmConfiguration(DeleteHsmConfigurationRequest request) {request = beforeClientExecution(request);return executeDeleteHsmConfiguration(request);}
public CreateLoadBalancerRequest(String loadBalancerName) {setLoadBalancerName(loadBalancerName);}
public String getUserInfo() {return decode(userInfo);}
public TagAttendeeResult tagAttendee(TagAttendeeRequest request) {request = beforeClientExecution(request);return executeTagAttendee(request);}
public String getRefName() {return name;}
public SpanNearQuery build() {return new SpanNearQuery(clauses.toArray(new SpanQuery[clauses.size()]), slop, ordered);}
public boolean isSubTotal(int rowIndex, int columnIndex) {return false;}
public DescribeDBProxiesResult describeDBProxies(DescribeDBProxiesRequest request) {request = beforeClientExecution(request);return executeDescribeDBProxies(request);}
public GetVoiceConnectorProxyResult getVoiceConnectorProxy(GetVoiceConnectorProxyRequest request) {request = beforeClientExecution(request);return executeGetVoiceConnectorProxy(request);}
public WindowCacheConfig fromConfig(Config rc) {setPackedGitUseStrongRefs(rc.getBoolean(CONFIG_CORE_SECTION,CONFIG_KEY_PACKED_GIT_USE_STRONGREFS,isPackedGitUseStrongRefs()));setPackedGitOpenFiles(rc.getInt(CONFIG_CORE_SECTION, null,CONFIG_KEY_PACKED_GIT_OPENFILES, getPackedGitOpenFiles()));setPackedGitLimit(rc.getLong(CONFIG_CORE_SECTION, null,CONFIG_KEY_PACKED_GIT_LIMIT, getPackedGitLimit()));setPackedGitWindowSize(rc.getInt(CONFIG_CORE_SECTION, null,CONFIG_KEY_PACKED_GIT_WINDOWSIZE, getPackedGitWindowSize()));setPackedGitMMAP(rc.getBoolean(CONFIG_CORE_SECTION, null,CONFIG_KEY_PACKED_GIT_MMAP, isPackedGitMMAP()));setDeltaBaseCacheLimit(rc.getInt(CONFIG_CORE_SECTION, null,CONFIG_KEY_DELTA_BASE_CACHE_LIMIT, getDeltaBaseCacheLimit()));long maxMem = Runtime.getRuntime().maxMemory();long sft = rc.getLong(CONFIG_CORE_SECTION, null,CONFIG_KEY_STREAM_FILE_TRESHOLD, getStreamFileThreshold());sft = Math.min(sft, maxMem / 4); sft = Math.min(sft, Integer.MAX_VALUE); setStreamFileThreshold((int) sft);return this;}
public static Date getJavaDate(double date) {return getJavaDate(date, false, null, false);}
public StartPersonTrackingResult startPersonTracking(StartPersonTrackingRequest request) {request = beforeClientExecution(request);return executeStartPersonTracking(request);}
@Override public int size() {return totalSize;}
public GetRouteResult getRoute(GetRouteRequest request) {request = beforeClientExecution(request);return executeGetRoute(request);}
public DeleteClusterResult deleteCluster(DeleteClusterRequest request) {request = beforeClientExecution(request);return executeDeleteCluster(request);}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append("[MMS]\n");buffer.append("    .addMenu        = ").append(Integer.toHexString(getAddMenuCount())).append("\n");buffer.append("    .delMenu        = ").append(Integer.toHexString(getDelMenuCount())).append("\n");buffer.append("[/MMS]\n");return buffer.toString();}
public FileBasedConfig(Config base, File cfgLocation, FS fs) {super(base);configFile = cfgLocation;this.fs = fs;this.snapshot = FileSnapshot.DIRTY;this.hash = ObjectId.zeroId();}
public int following(int pos) {if (pos < text.getBeginIndex() || pos > text.getEndIndex()) {throw new IllegalArgumentException("offset out of bounds");} else if (0 == sentenceStarts.length) {text.setIndex(text.getBeginIndex());return DONE;} else if (pos >= sentenceStarts[sentenceStarts.length - 1]) {text.setIndex(text.getEndIndex());currentSentence = sentenceStarts.length - 1;return DONE;} else { currentSentence = (sentenceStarts.length - 1) / 2; moveToSentenceAt(pos, 0, sentenceStarts.length - 2);text.setIndex(sentenceStarts[++currentSentence]);return current();}}
public UpdateParameterGroupResult updateParameterGroup(UpdateParameterGroupRequest request) {request = beforeClientExecution(request);return executeUpdateParameterGroup(request);}
public SeriesChartGroupIndexRecord clone() {return copy();}
public static double calcDistanceFromErrPct(Shape shape, double distErrPct, SpatialContext ctx) {if (distErrPct < 0 || distErrPct > 0.5) {throw new IllegalArgumentException("distErrPct " + distErrPct + " must be between [0 to 0.5]");}if (distErrPct == 0 || shape instanceof Point) {return 0;}Rectangle bbox = shape.getBoundingBox();Point ctr = bbox.getCenter();double y = (ctr.getY() >= 0 ? bbox.getMaxY() : bbox.getMinY());double diagonalDist = ctx.getDistCalc().distance(ctr, bbox.getMaxX(), y);return diagonalDist * distErrPct;}
public int codePointAt(int index) {if (index < 0 || index >= count) {throw indexAndLength(index);}return Character.codePointAt(value, index, count);}
public void setPasswordVerifier(int passwordVerifier) {this.passwordVerifier = passwordVerifier;}
public ListVaultsRequest(String accountId) {setAccountId(accountId);}
public SquashMessageFormatter() {dateFormatter = new GitDateFormatter(Format.DEFAULT);}
public GetVideoCoverRequest() {super("CloudPhoto", "2017-07-11", "GetVideoCover", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public int lastIndexOf(Object object) {int pos = size;Link<E> link = voidLink.previous;if (object != null) {while (link != voidLink) {pos--;if (object.equals(link.data)) {return pos;}link = link.previous;}} else {while (link != voidLink) {pos--;if (link.data == null) {return pos;}link = link.previous;}}return -1;}
public DescribeSpotFleetRequestsResult describeSpotFleetRequests(DescribeSpotFleetRequestsRequest request) {request = beforeClientExecution(request);return executeDescribeSpotFleetRequests(request);}
public IndexFacesResult indexFaces(IndexFacesRequest request) {request = beforeClientExecution(request);return executeIndexFaces(request);}
public RuleBasedBreakIterator getBreakIterator(int script) {switch(script) {case UScript.JAPANESE: return (RuleBasedBreakIterator)cjkBreakIterator.clone();case UScript.MYANMAR:if (myanmarAsWords) {return (RuleBasedBreakIterator)defaultBreakIterator.clone();} else {return (RuleBasedBreakIterator)myanmarSyllableIterator.clone();}default: return (RuleBasedBreakIterator)defaultBreakIterator.clone();}}
public String toString(){StringBuilder b = new StringBuilder();b.append("[DCONREF]\n");b.append("    .ref\n");b.append("        .firstrow   = ").append(firstRow).append("\n");b.append("        .lastrow    = ").append(lastRow).append("\n");b.append("        .firstcol   = ").append(firstCol).append("\n");b.append("        .lastcol    = ").append(lastCol).append("\n");b.append("    .cch            = ").append(charCount).append("\n");b.append("    .stFile\n");b.append("        .h          = ").append(charType).append("\n");b.append("        .rgb        = ").append(getReadablePath()).append("\n");b.append("[/DCONREF]\n");return b.toString();}
public int getPackedGitOpenFiles() {return packedGitOpenFiles;}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[FEATURE HEADER]\n");buffer.append("[/FEATURE HEADER]\n");return buffer.toString();}
public static byte[] getToUnicodeLE(String string) {return string.getBytes(UTF16LE);}
public final List<String> getFooterLines(String keyName) {return getFooterLines(new FooterKey(keyName));}
public void refresh() {super.refresh();clearReferences();}
public float get(int index) {checkIndex(index);return byteBuffer.getFloat(index * SizeOf.FLOAT);}
public DeleteDetectorResult deleteDetector(DeleteDetectorRequest request) {request = beforeClientExecution(request);return executeDeleteDetector(request);}
public int[] grow() {assert bytesStart != null;return bytesStart = ArrayUtil.grow(bytesStart, bytesStart.length + 1);}
public ListExclusionsResult listExclusions(ListExclusionsRequest request) {request = beforeClientExecution(request);return executeListExclusions(request);}
public static SpatialStrategy getSpatialStrategy(int roundNumber) {SpatialStrategy result = spatialStrategyCache.get(roundNumber);if (result == null) {throw new IllegalStateException("Strategy should have been init'ed by SpatialDocMaker by now");}return result;}
public DBCluster restoreDBClusterToPointInTime(RestoreDBClusterToPointInTimeRequest request) {request = beforeClientExecution(request);return executeRestoreDBClusterToPointInTime(request);}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_categoryDataType);out.writeShort(field_2_valuesDataType);out.writeShort(field_3_numCategories);out.writeShort(field_4_numValues);out.writeShort(field_5_bubbleSeriesType);out.writeShort(field_6_numBubbleValues);}
public PostAgentProfileResult postAgentProfile(PostAgentProfileRequest request) {request = beforeClientExecution(request);return executePostAgentProfile(request);}
public ParseTreePattern compileParseTreePattern(String pattern, int patternRuleIndex) {if ( getTokenStream()!=null ) {TokenSource tokenSource = getTokenStream().getTokenSource();if ( tokenSource instanceof Lexer ) {Lexer lexer = (Lexer)tokenSource;return compileParseTreePattern(pattern, patternRuleIndex, lexer);}}throw new UnsupportedOperationException("Parser can't discover a lexer to use");}
public BacktrackDBClusterResult backtrackDBCluster(BacktrackDBClusterRequest request) {request = beforeClientExecution(request);return executeBacktrackDBCluster(request);}
public String getName() {return strategyName;}
public void copyTo(byte[] b, int o) {formatHexByte(b, o + 0, w1);formatHexByte(b, o + 8, w2);formatHexByte(b, o + 16, w3);formatHexByte(b, o + 24, w4);formatHexByte(b, o + 32, w5);}
public static final IntList lineMap(byte[] buf, int ptr, int end) {IntList map = new IntList((end - ptr) / 36);map.fillTo(1, Integer.MIN_VALUE);for (; ptr < end; ptr = nextLF(buf, ptr)) {map.add(ptr);}map.add(end);return map;}
public Set<ObjectId> getAdditionalHaves() {return Collections.emptySet();}
public synchronized long ramBytesUsed() {long sizeInBytes = BASE_RAM_BYTES_USED + fields.size() * 2 * RamUsageEstimator.NUM_BYTES_OBJECT_REF;for(SimpleTextTerms simpleTextTerms : termsCache.values()) {sizeInBytes += (simpleTextTerms!=null) ? simpleTextTerms.ramBytesUsed() : 0;}return sizeInBytes;}
public String toXml(String tab) {StringBuilder builder = new StringBuilder();builder.append(tab).append("<").append(getRecordName()).append(">\n");for (EscherRecord escherRecord : getEscherRecords()) {builder.append(escherRecord.toXml(tab + "\t"));}builder.append(tab).append("</").append(getRecordName()).append(">\n");return builder.toString();}
public TokenStream create(TokenStream input) {return new GalicianMinimalStemFilter(input);}
public String toString() {StringBuilder r = new StringBuilder();r.append("Commit");r.append("={\n");r.append("tree ");r.append(treeId != null ? treeId.name() : "NOT_SET");r.append("\n");for (ObjectId p : parentIds) {r.append("parent ");r.append(p.name());r.append("\n");}r.append("author ");r.append(author != null ? author.toString() : "NOT_SET");r.append("\n");r.append("committer ");r.append(committer != null ? committer.toString() : "NOT_SET");r.append("\n");r.append("gpgSignature ");r.append(gpgSignature != null ? gpgSignature.toString() : "NOT_SET");r.append("\n");if (encoding != null && !References.isSameObject(encoding, UTF_8)) {r.append("encoding ");r.append(encoding.name());r.append("\n");}r.append("\n");r.append(message != null ? message : "");r.append("}");return r.toString();}
public IndicNormalizationFilterFactory(Map<String,String> args) {super(args);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public OptionGroup createOptionGroup(CreateOptionGroupRequest request) {request = beforeClientExecution(request);return executeCreateOptionGroup(request);}
public AssociateMemberAccountResult associateMemberAccount(AssociateMemberAccountRequest request) {request = beforeClientExecution(request);return executeAssociateMemberAccount(request);}
public void run() {doRefreshProgress(mId, mProgress, mFromUser, true);mRefreshProgressRunnable = this;}
public SetTerminationProtectionResult setTerminationProtection(SetTerminationProtectionRequest request) {request = beforeClientExecution(request);return executeSetTerminationProtection(request);}
public String getErrorHeader(RecognitionException e) {int line = e.getOffendingToken().getLine();int charPositionInLine = e.getOffendingToken().getCharPositionInLine();return "line "+line+":"+charPositionInLine;}
public CharBuffer asReadOnlyBuffer() {CharToByteBufferAdapter buf = new CharToByteBufferAdapter(byteBuffer.asReadOnlyBuffer());buf.limit = limit;buf.position = position;buf.mark = mark;buf.byteBuffer.order = byteBuffer.order;return buf;}
public StopSentimentDetectionJobResult stopSentimentDetectionJob(StopSentimentDetectionJobRequest request) {request = beforeClientExecution(request);return executeStopSentimentDetectionJob(request);}
public ObjectIdSubclassMap<ObjectId> getNewObjectIds() {if (newObjectIds != null)return newObjectIds;return new ObjectIdSubclassMap<>();}
public void clear() {hash = hash(new byte[0]);super.clear();}
public void reset() throws IOException {synchronized (lock) {checkNotClosed();if (mark == -1) {throw new IOException("Invalid mark");}pos = mark;}}
public RefErrorPtg(LittleEndianInput in)  {field_1_reserved = in.readInt();}
public SuspendGameServerGroupResult suspendGameServerGroup(SuspendGameServerGroupRequest request) {request = beforeClientExecution(request);return executeSuspendGameServerGroup(request);}
public final ValueEval evaluate(ValueEval[] args, int srcRowIndex, int srcColumnIndex) {if (args.length != 3) {return ErrorEval.VALUE_INVALID;}return evaluate(srcRowIndex, srcColumnIndex, args[0], args[1], args[2]);}
public GetRepoRequest() {super("cr", "2016-06-07", "GetRepo", "cr");setUriPattern("/repos/[RepoNamespace]/[RepoName]");setMethod(MethodType.GET);}
public void setDate(Date date) {if (date != null) {setDate(DateTools.dateToString(date, DateTools.Resolution.SECOND));} else {this.date = null;}}
public TokenStream create(TokenStream input) {return new GermanMinimalStemFilter(input);}
public Object[] toArray() {return a.clone();}
public void write(char[] buffer, int offset, int len) {Arrays.checkOffsetAndCount(buffer.length, offset, len);synchronized (lock) {expand(len);System.arraycopy(buffer, offset, this.buf, this.count, len);this.count += len;}}
public static final RevFilter after(Date ts) {return after(ts.getTime());}
public DeleteGroupPolicyRequest(String groupName, String policyName) {setGroupName(groupName);setPolicyName(policyName);}
public DeregisterTransitGatewayMulticastGroupMembersResult deregisterTransitGatewayMulticastGroupMembers(DeregisterTransitGatewayMulticastGroupMembersRequest request) {request = beforeClientExecution(request);return executeDeregisterTransitGatewayMulticastGroupMembers(request);}
public BatchDeleteScheduledActionResult batchDeleteScheduledAction(BatchDeleteScheduledActionRequest request) {request = beforeClientExecution(request);return executeBatchDeleteScheduledAction(request);}
public CreateAlgorithmResult createAlgorithm(CreateAlgorithmRequest request) {request = beforeClientExecution(request);return executeCreateAlgorithm(request);}
public int readUByte() {return readByte() & 0x00FF;}
public void setLength(int sz) {NB.encodeInt32(info, infoOffset + P_SIZE, sz);}
public DescribeScalingProcessTypesResult describeScalingProcessTypes() {return describeScalingProcessTypes(new DescribeScalingProcessTypesRequest());}
public ListResourceRecordSetsResult listResourceRecordSets(ListResourceRecordSetsRequest request) {request = beforeClientExecution(request);return executeListResourceRecordSets(request);}
public Token recoverInline(Parser recognizer)throws RecognitionException{InputMismatchException e = new InputMismatchException(recognizer);for (ParserRuleContext context = recognizer.getContext(); context != null; context = context.getParent()) {context.exception = e;}throw new ParseCancellationException(e);}
public SetTagsForResourceResult setTagsForResource(SetTagsForResourceRequest request) {request = beforeClientExecution(request);return executeSetTagsForResource(request);}
public ModifyStrategyRequest() {super("CloudCallCenter", "2017-07-05", "ModifyStrategy", "CloudCallCenter", "innerAPI");}
public DescribeVpcEndpointServicesResult describeVpcEndpointServices(DescribeVpcEndpointServicesRequest request) {request = beforeClientExecution(request);return executeDescribeVpcEndpointServices(request);}
public EnableLoggingResult enableLogging(EnableLoggingRequest request) {request = beforeClientExecution(request);return executeEnableLogging(request);}
public boolean contains(Object o) {return ConcurrentHashMap.this.containsValue(o);}
public SheetRangeIdentifier(String bookName, NameIdentifier firstSheetIdentifier, NameIdentifier lastSheetIdentifier) {super(bookName, firstSheetIdentifier);_lastSheetIdentifier = lastSheetIdentifier;}
public DomainMetadataRequest(String domainName) {setDomainName(domainName);}
public ParseException(Token currentTokenVal,int[][] expectedTokenSequencesVal, String[] tokenImageVal) {super(new MessageImpl(QueryParserMessages.INVALID_SYNTAX, initialise(currentTokenVal, expectedTokenSequencesVal, tokenImageVal)));this.currentToken = currentTokenVal;this.expectedTokenSequences = expectedTokenSequencesVal;this.tokenImage = tokenImageVal;}
public FetchPhotosRequest() {super("CloudPhoto", "2017-07-11", "FetchPhotos", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public PrintWriter writer() {return writer;}
public NGramTokenizerFactory(Map<String, String> args) {super(args);minGramSize = getInt(args, "minGramSize", NGramTokenizer.DEFAULT_MIN_NGRAM_SIZE);maxGramSize = getInt(args, "maxGramSize", NGramTokenizer.DEFAULT_MAX_NGRAM_SIZE);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public boolean isDirectoryFileConflict() {return dfConflict != null;}
public IndonesianStemFilter(TokenStream input, boolean stemDerivational) {super(input);this.stemDerivational = stemDerivational;}
public CreateTrafficPolicyResult createTrafficPolicy(CreateTrafficPolicyRequest request) {request = beforeClientExecution(request);return executeCreateTrafficPolicy(request);}
public void serialize(LittleEndianOutput out) {out.writeInt(fSD);out.writeInt(passwordVerifier);StringUtil.writeUnicodeString(out, title);out.write(securityDescriptor);}
public static double floor(double n, double s) {if (s==0 && n!=0) {return Double.NaN;} else {return (n==0 || s==0) ? 0 : Math.floor(n/s) * s;}}
public ByteArrayDataOutput(byte[] bytes, int offset, int len) {reset(bytes, offset, len);}
public static List<Tree> getChildren(Tree t) {List<Tree> kids = new ArrayList<Tree>();for (int i=0; i<t.getChildCount(); i++) {kids.add(t.getChild(i));}return kids;}
public void clear() {Hashtable.this.clear();}
public RefreshAllRecord(boolean refreshAll) {this(0);setRefreshAll(refreshAll);}
public DeleteNamedQueryResult deleteNamedQuery(DeleteNamedQueryRequest request) {request = beforeClientExecution(request);return executeDeleteNamedQuery(request);}
public GraphvizFormatter(ConnectionCosts costs) {this.costs = costs;this.bestPathMap = new HashMap<>();sb.append(formatHeader());sb.append("  init [style=invis]\n");sb.append("  init -> 0.0 [label=\"" + BOS_LABEL + "\"]\n");}
public CheckMultiagentRequest() {super("visionai-poc", "2020-04-08", "CheckMultiagent");setMethod(MethodType.POST);}
public ListUserProfilesResult listUserProfiles(ListUserProfilesRequest request) {request = beforeClientExecution(request);return executeListUserProfiles(request);}
public CreateRelationalDatabaseFromSnapshotResult createRelationalDatabaseFromSnapshot(CreateRelationalDatabaseFromSnapshotRequest request) {request = beforeClientExecution(request);return executeCreateRelationalDatabaseFromSnapshot(request);}
public StartTaskResult startTask(StartTaskRequest request) {request = beforeClientExecution(request);return executeStartTask(request);}
public Set<String> getIgnoredPaths() {return ignoredPaths;}
public FeatSmartTag(RecordInputStream in) {data = in.readRemainder();}
public Change(ChangeAction action, ResourceRecordSet resourceRecordSet) {setAction(action.toString());setResourceRecordSet(resourceRecordSet);}
public DeleteImageResult deleteImage(DeleteImageRequest request) {request = beforeClientExecution(request);return executeDeleteImage(request);}
public CreateConfigurationSetResult createConfigurationSet(CreateConfigurationSetRequest request) {request = beforeClientExecution(request);return executeCreateConfigurationSet(request);}
public Iterator<E> iterator() {Object[] snapshot = elements;return new CowIterator<E>(snapshot, 0, snapshot.length);}
public void visitContainedRecords(RecordVisitor rv) {if (_recs.isEmpty()) {return;}rv.visitRecord(_bofRec);for (int i = 0; i < _recs.size(); i++) {RecordBase rb = _recs.get(i);if (rb instanceof RecordAggregate) {((RecordAggregate) rb).visitContainedRecords(rv);} else {rv.visitRecord((org.apache.poi.hssf.record.Record) rb);}}rv.visitRecord(EOFRecord.instance);}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append("[FtCbls ]").append("\n");buffer.append("  size     = ").append(getDataSize()).append("\n");buffer.append("  reserved = ").append(HexDump.toHex(reserved)).append("\n");buffer.append("[/FtCbls ]").append("\n");return buffer.toString();}
public static BATBlock createEmptyBATBlock(final POIFSBigBlockSize bigBlockSize, boolean isXBAT) {BATBlock block = new BATBlock(bigBlockSize);if(isXBAT) {final int _entries_per_xbat_block = bigBlockSize.getXBATEntriesPerBlock();block._values[ _entries_per_xbat_block ] = POIFSConstants.END_OF_CHAIN;}return block;}
public TagResourceResult tagResource(TagResourceRequest request) {request = beforeClientExecution(request);return executeTagResource(request);}
public DeleteMailboxPermissionsResult deleteMailboxPermissions(DeleteMailboxPermissionsRequest request) {request = beforeClientExecution(request);return executeDeleteMailboxPermissions(request);}
public ListDatasetGroupsResult listDatasetGroups(ListDatasetGroupsRequest request) {request = beforeClientExecution(request);return executeListDatasetGroups(request);}
public ResumeProcessesResult resumeProcesses(ResumeProcessesRequest request) {request = beforeClientExecution(request);return executeResumeProcesses(request);}
public GetPersonTrackingResult getPersonTracking(GetPersonTrackingRequest request) {request = beforeClientExecution(request);return executeGetPersonTracking(request);}
public String toFormulaString(String[] operands) {if(space.isSet(_options)) {return operands[ 0 ];} else if (optiIf.isSet(_options)) {return toFormulaString() + "(" + operands[0] + ")";} else if (optiSkip.isSet(_options)) {return toFormulaString() + operands[0];   } else {return toFormulaString() + "(" + operands[0] + ")";}}
public T merge(T first, T second) {throw new UnsupportedOperationException();}
public String toString() {return this.message.getKey() + ": " + getLocalizedMessage();}
public XPath(Parser parser, String path) {this.parser = parser;this.path = path;elements = split(path);}
public CreateAccountAliasRequest(String accountAlias) {setAccountAlias(accountAlias);}
public void decode(byte[] blocks, int blocksOffset, long[] values, int valuesOffset, int iterations) {for (int j = 0; j < iterations; ++j) {final byte block = blocks[blocksOffset++];values[valuesOffset++] = (block >>> 7) & 1;values[valuesOffset++] = (block >>> 6) & 1;values[valuesOffset++] = (block >>> 5) & 1;values[valuesOffset++] = (block >>> 4) & 1;values[valuesOffset++] = (block >>> 3) & 1;values[valuesOffset++] = (block >>> 2) & 1;values[valuesOffset++] = (block >>> 1) & 1;values[valuesOffset++] = block & 1;}}
public PushConnection openPush() throws TransportException {return new TcpPushConnection();}
public static void strcpy(char[] dst, int di, char[] src, int si) {while (src[si] != 0) {dst[di++] = src[si++];}dst[di] = 0;}
@Override public K getKey() {return mapEntry.getKey();}
public static int numNonnull(Object[] data) {int n = 0;if ( data == null ) return n;for (Object o : data) {if ( o!=null ) n++;}return n;}
public void add(int location, E object) {if (location >= 0 && location <= size) {Link<E> link = voidLink;if (location < (size / 2)) {for (int i = 0; i <= location; i++) {link = link.next;}} else {for (int i = size; i > location; i--) {link = link.previous;}}Link<E> previous = link.previous;Link<E> newLink = new Link<E>(object, previous, link);previous.next = newLink;link.previous = newLink;size++;modCount++;} else {throw new IndexOutOfBoundsException();}}
public DescribeDomainResult describeDomain(DescribeDomainRequest request) {request = beforeClientExecution(request);return executeDescribeDomain(request);}
public void flush() throws IOException {super.flush();}
public PersianCharFilterFactory(Map<String,String> args) {super(args);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public boolean incrementToken() {if (used) {return false;}clearAttributes();termAttribute.append(value);offsetAttribute.setOffset(0, length);used = true;return true;}
public static FloatBuffer allocate(int capacity) {if (capacity < 0) {throw new IllegalArgumentException();}return new ReadWriteFloatArrayBuffer(capacity);}
public final Edit after(Edit cut) {return new Edit(cut.endA, endA, cut.endB, endB);}
public UpdateRuleVersionResult updateRuleVersion(UpdateRuleVersionRequest request) {request = beforeClientExecution(request);return executeUpdateRuleVersion(request);}
public ListVoiceConnectorTerminationCredentialsResult listVoiceConnectorTerminationCredentials(ListVoiceConnectorTerminationCredentialsRequest request) {request = beforeClientExecution(request);return executeListVoiceConnectorTerminationCredentials(request);}
public GetDeploymentTargetResult getDeploymentTarget(GetDeploymentTargetRequest request) {request = beforeClientExecution(request);return executeGetDeploymentTarget(request);}
public void setNoChildReport() {letChildReport  = false;for (final PerfTask task : tasks) {if (task instanceof TaskSequence) {((TaskSequence)task).setNoChildReport();}}}
public E get(int location) {try {return a[location];} catch (ArrayIndexOutOfBoundsException e) {throw java.util.ArrayList.throwIndexOutOfBoundsException(location, a.length);}}
public DescribeDataSetResult describeDataSet(DescribeDataSetRequest request) {request = beforeClientExecution(request);return executeDescribeDataSet(request);}
public SkipWorkTreeFilter(int treeIdx) {this.treeIdx = treeIdx;}
public DescribeNetworkInterfacesResult describeNetworkInterfaces() {return describeNetworkInterfaces(new DescribeNetworkInterfacesRequest());}
public final boolean contains(int row, int col) {return _firstRow <= row && _lastRow >= row&& _firstColumn <= col && _lastColumn >= col;}
public String toString() {return new String(this.chars);}
public PatchType getPatchType() {return patchType;}
public Iterator<K> iterator() {return new KeyIterator();}
public CreateScriptResult createScript(CreateScriptRequest request) {request = beforeClientExecution(request);return executeCreateScript(request);}
public BytesRef next() {termUpto++;if (termUpto >= info.terms.size()) {return null;} else {info.terms.get(info.sortedTerms[termUpto], br);return br;}}
public String outputToString(CharsRef output) {return output.toString();}
public AssociateWebsiteAuthorizationProviderResult associateWebsiteAuthorizationProvider(AssociateWebsiteAuthorizationProviderRequest request) {request = beforeClientExecution(request);return executeAssociateWebsiteAuthorizationProvider(request);}
public void unpop(RevCommit c) {Block b = head;if (b == null) {b = free.newBlock();b.resetToMiddle();b.add(c);head = b;tail = b;return;} else if (b.canUnpop()) {b.unpop(c);return;}b = free.newBlock();b.resetToEnd();b.unpop(c);b.next = head;head = b;}
public EdgeNGramTokenizerFactory(Map<String, String> args) {super(args);minGramSize = getInt(args, "minGramSize", EdgeNGramTokenizer.DEFAULT_MIN_GRAM_SIZE);maxGramSize = getInt(args, "maxGramSize", EdgeNGramTokenizer.DEFAULT_MAX_GRAM_SIZE);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public ModifyDBParameterGroupRequest(String dBParameterGroupName, java.util.List<Parameter> parameters) {setDBParameterGroupName(dBParameterGroupName);setParameters(parameters);}
public GetHostedZoneLimitResult getHostedZoneLimit(GetHostedZoneLimitRequest request) {request = beforeClientExecution(request);return executeGetHostedZoneLimit(request);}
public void set(int index, long value) {final int o = index >>> 6;final int b = index & 63;final int shift = b << 0;blocks[o] = (blocks[o] & ~(1L << shift)) | (value << shift);}
public RevFilter clone() {return new PatternSearch(pattern());}
public String toString() {return "spans(" + term.toString() + ")@" +(doc == -1 ? "START" : (doc == NO_MORE_DOCS) ? "ENDDOC": doc + " - " + (position == NO_MORE_POSITIONS ? "ENDPOS" : position));}
public boolean canAppendMatch() {for (Head head : heads) {if (head != LastHead.INSTANCE) {return true;}}return false;}
public synchronized int lastIndexOf(String subString, int start) {return super.lastIndexOf(subString, start);}
public DeleteNetworkAclEntryResult deleteNetworkAclEntry(DeleteNetworkAclEntryRequest request) {request = beforeClientExecution(request);return executeDeleteNetworkAclEntry(request);}
public AssociateMemberToGroupResult associateMemberToGroup(AssociateMemberToGroupRequest request) {request = beforeClientExecution(request);return executeAssociateMemberToGroup(request);}
public static final int committer(byte[] b, int ptr) {final int sz = b.length;if (ptr == 0)ptr += 46; while (ptr < sz && b[ptr] == 'p')ptr += 48; if (ptr < sz && b[ptr] == 'a')ptr = nextLF(b, ptr);return match(b, ptr, committer);}
public int getLineNumber() { return row; }
public SubmoduleUpdateCommand addPath(String path) {paths.add(path);return this;}
public GetPushTemplateResult getPushTemplate(GetPushTemplateRequest request) {request = beforeClientExecution(request);return executeGetPushTemplate(request);}
public DescribeVaultResult describeVault(DescribeVaultRequest request) {request = beforeClientExecution(request);return executeDescribeVault(request);}
public DescribeVpcPeeringConnectionsResult describeVpcPeeringConnections() {return describeVpcPeeringConnections(new DescribeVpcPeeringConnectionsRequest());}
public ByteBuffer putLong(int index, long value) {throw new ReadOnlyBufferException();}
public RegisterDeviceResult registerDevice(RegisterDeviceRequest request) {request = beforeClientExecution(request);return executeRegisterDevice(request);}
public static Format byId(int id) {for (Format format : Format.values()) {if (format.getId() == id) {return format;}}throw new IllegalArgumentException("Unknown format id: " + id);}
public DeleteAppResult deleteApp(DeleteAppRequest request) {request = beforeClientExecution(request);return executeDeleteApp(request);}
public GetBaiduChannelResult getBaiduChannel(GetBaiduChannelRequest request) {request = beforeClientExecution(request);return executeGetBaiduChannel(request);}
public FST.BytesReader getBytesReader() {return fst.getBytesReader();}
public static boolean isValidSchemeChar(int index, char c) {if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {return true;}if (index > 0 && ((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.')) {return true;}return false;}
public ListAppliedSchemaArnsResult listAppliedSchemaArns(ListAppliedSchemaArnsRequest request) {request = beforeClientExecution(request);return executeListAppliedSchemaArns(request);}
public String name() {return this.name;}
public ValueEval evaluate(ValueEval[] args, OperationEvaluationContext ec) {if (args.length < 1) {return ErrorEval.VALUE_INVALID;}boolean isA1style;String text;try {ValueEval ve = OperandResolver.getSingleValue(args[0], ec.getRowIndex(), ec.getColumnIndex());text = OperandResolver.coerceValueToString(ve);switch (args.length) {case 1:isA1style = true;break;case 2:isA1style = evaluateBooleanArg(args[1], ec);break;default:return ErrorEval.VALUE_INVALID;}} catch (EvaluationException e) {return e.getErrorEval();}return evaluateIndirect(ec, text, isA1style);}
public final int compareTo(int[] bs, int p) {int cmp;cmp = NB.compareUInt32(w1, bs[p]);if (cmp != 0)return cmp;cmp = NB.compareUInt32(w2, bs[p + 1]);if (cmp != 0)return cmp;cmp = NB.compareUInt32(w3, bs[p + 2]);if (cmp != 0)return cmp;cmp = NB.compareUInt32(w4, bs[p + 3]);if (cmp != 0)return cmp;return NB.compareUInt32(w5, bs[p + 4]);}
public void removeName(int index){names.remove(index);workbook.removeName(index);}
public GetQueueAttributesRequest(String queueUrl, java.util.List<String> attributeNames) {setQueueUrl(queueUrl);setAttributeNames(attributeNames);}
public static boolean[] copyOf(boolean[] original, int newLength) {if (newLength < 0) {throw new NegativeArraySizeException();}return copyOfRange(original, 0, newLength);}
public static void setEnabled(boolean enabled) {ENABLED = enabled;}
public DeleteLogPatternResult deleteLogPattern(DeleteLogPatternRequest request) {request = beforeClientExecution(request);return executeDeleteLogPattern(request);}
public boolean contains(char[] text, int off, int len) {return map.containsKey(text, off, len);}
public int getFirstSheetIndexFromExternSheetIndex(int externSheetNumber){return linkTable.getFirstInternalSheetIndexForExtIndex(externSheetNumber);}
public boolean handles(String commandLine) {return command.length() + 1 < commandLine.length()&& commandLine.charAt(command.length()) == ' '&& commandLine.startsWith(command);}
public static void register(MergeStrategy imp) {register(imp.getName(), imp);}
public long ramBytesUsed() {return BASE_RAM_BYTES_USED + ((index!=null)? index.ramBytesUsed() : 0);}
public HostedZone(String id, String name, String callerReference) {setId(id);setName(name);setCallerReference(callerReference);}
public GetFindingsResult getFindings(GetFindingsRequest request) {request = beforeClientExecution(request);return executeGetFindings(request);}
public DescribeTopicsDetectionJobResult describeTopicsDetectionJob(DescribeTopicsDetectionJobRequest request) {request = beforeClientExecution(request);return executeDescribeTopicsDetectionJob(request);}
public boolean processMatch(ValueEval eval) {if(eval instanceof NumericValueEval) {if(minimumValue == null) { minimumValue = eval;} else { double currentValue = ((NumericValueEval)eval).getNumberValue();double oldValue = ((NumericValueEval)minimumValue).getNumberValue();if(currentValue < oldValue) {minimumValue = eval;}}}return true;}
public void write(LittleEndianOutput out) {out.writeByte(sid + getPtgClass());out.writeShort(field_1_len_ref_subexpression);}
public static void main(String[] args) throws IOException {boolean printTree = false;String path = null;for(int i=0;i<args.length;i++) {if (args[i].equals("-printTree")) {printTree = true;} else {path = args[i];}}if (args.length != (printTree ? 2 : 1)) {System.out.println("\nUsage: java -classpath ... org.apache.lucene.facet.util.PrintTaxonomyStats [-printTree] /path/to/taxononmy/index\n");System.exit(1);}Directory dir = FSDirectory.open(Paths.get(path));TaxonomyReader r = new DirectoryTaxonomyReader(dir);printStats(r, System.out, printTree);r.close();dir.close();}
public void setByteValue(byte value) {if (!(fieldsData instanceof Byte)) {throw new IllegalArgumentException("cannot change value type from " + fieldsData.getClass().getSimpleName() + " to Byte");}fieldsData = Byte.valueOf(value);}
public static int initialize() {return initialize(DEFAULT_SEED);}
public CachingDoubleValueSource(DoubleValuesSource source) {this.source = source;cache = new HashMap<>();}
public AttributeDefinition(String attributeName, ScalarAttributeType attributeType) {setAttributeName(attributeName);setAttributeType(attributeType.toString());}
public static String join(Collection<String> parts, String separator) {return StringUtils.join(parts, separator, separator);}
public ListTaskDefinitionFamiliesResult listTaskDefinitionFamilies(ListTaskDefinitionFamiliesRequest request) {request = beforeClientExecution(request);return executeListTaskDefinitionFamilies(request);}
public ListComponentsResult listComponents(ListComponentsRequest request) {request = beforeClientExecution(request);return executeListComponents(request);}
public ActivatePhotosRequest() {super("CloudPhoto", "2017-07-11", "ActivatePhotos", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public CreateMatchmakingRuleSetResult createMatchmakingRuleSet(CreateMatchmakingRuleSetRequest request) {request = beforeClientExecution(request);return executeCreateMatchmakingRuleSet(request);}
public ListAvailableManagementCidrRangesResult listAvailableManagementCidrRanges(ListAvailableManagementCidrRangesRequest request) {request = beforeClientExecution(request);return executeListAvailableManagementCidrRanges(request);}
public ObjectIdSubclassMap<ObjectId> getBaseObjectIds() {if (baseObjectIds != null)return baseObjectIds;return new ObjectIdSubclassMap<>();}
public DeletePushTemplateResult deletePushTemplate(DeletePushTemplateRequest request) {request = beforeClientExecution(request);return executeDeletePushTemplate(request);}
public CreateDomainEntryResult createDomainEntry(CreateDomainEntryRequest request) {request = beforeClientExecution(request);return executeCreateDomainEntry(request);}
public static int getEncodedSize(Object[] values) {int result = values.length * 1;for (Object value : values) {result += getEncodedSize(value);}return result;}
public OpenNLPTokenizerFactory(Map<String,String> args) {super(args);sentenceModelFile = require(args, SENTENCE_MODEL);tokenizerModelFile = require(args, TOKENIZER_MODEL);if ( ! args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public final int getInt(int index) {checkIndex(index, SizeOf.INT);return Memory.peekInt(backingArray, offset + index, order);}
public List<Head> getNextHeads(char c) {if (matches(c)) {return newHeads;}return FileNameMatcher.EMPTY_HEAD_LIST;}
public ByteBuffer putShort(short value) {throw new ReadOnlyBufferException();}
public void writeUnshared(Object object) throws IOException {writeObject(object, true);}
public int offsetByCodePoints(int index, int codePointOffset) {return Character.offsetByCodePoints(value, 0, count, index,codePointOffset);}
public static int getUniqueAlt(Collection<BitSet> altsets) {BitSet all = getAlts(altsets);if ( all.cardinality()==1 ) return all.nextSetBit(0);return ATN.INVALID_ALT_NUMBER;}
public Date getWhen() {return new Date(when);}
public RuleTagToken(String ruleName, int bypassTokenType, String label) {if (ruleName == null || ruleName.isEmpty()) {throw new IllegalArgumentException("ruleName cannot be null or empty.");}this.ruleName = ruleName;this.bypassTokenType = bypassTokenType;this.label = label;}
public DisableOrganizationAdminAccountResult disableOrganizationAdminAccount(DisableOrganizationAdminAccountRequest request) {request = beforeClientExecution(request);return executeDisableOrganizationAdminAccount(request);}
public CreateRoomResult createRoom(CreateRoomRequest request) {request = beforeClientExecution(request);return executeCreateRoom(request);}
public ReplicationGroup deleteReplicationGroup(DeleteReplicationGroupRequest request) {request = beforeClientExecution(request);return executeDeleteReplicationGroup(request);}
public final CharBuffer decode(ByteBuffer buffer) {try {return newDecoder().onMalformedInput(CodingErrorAction.REPLACE).onUnmappableCharacter(CodingErrorAction.REPLACE).decode(buffer);} catch (CharacterCodingException ex) {throw new Error(ex.getMessage(), ex);}}
public Distribution(String id, String status, String domainName) {setId(id);setStatus(status);setDomainName(domainName);}
public final double[] array() {return protectedArray();}
public DateWindow1904Record(RecordInputStream in) {field_1_window = in.readShort();}
public DeleteDBSnapshotRequest(String dBSnapshotIdentifier) {setDBSnapshotIdentifier(dBSnapshotIdentifier);}
public final ParserExtension getExtension(String key) {return this.extensions.get(key);}
public void inform(ResourceLoader loader) {try {if (chunkerModelFile != null) {OpenNLPOpsFactory.getChunkerModel(chunkerModelFile, loader);}} catch (IOException e) {throw new IllegalArgumentException(e);}}
public CompleteVaultLockResult completeVaultLock(CompleteVaultLockRequest request) {request = beforeClientExecution(request);return executeCompleteVaultLock(request);}
public final int[] getCharIntervals() {return points.clone();}
public long ramBytesUsed() {return values.ramBytesUsed()+ super.ramBytesUsed()+ Long.BYTES+ RamUsageEstimator.NUM_BYTES_OBJECT_REF;}
public RegisterInstancesWithLoadBalancerResult registerInstancesWithLoadBalancer(RegisterInstancesWithLoadBalancerRequest request) {request = beforeClientExecution(request);return executeRegisterInstancesWithLoadBalancer(request);}
public DescribeClusterUserKubeconfigRequest() {super("CS", "2015-12-15", "DescribeClusterUserKubeconfig", "csk");setUriPattern("/k8s/[ClusterId]/user_config");setMethod(MethodType.GET);}
public PrecisionRecord(RecordInputStream in) {field_1_precision = in.readShort();}
public void serialize(LittleEndianOutput out) {out.writeShort(getLeftRowGutter());out.writeShort(getTopColGutter());out.writeShort(getRowLevelMax());out.writeShort(getColLevelMax());}
public DeleteVirtualInterfaceResult deleteVirtualInterface(DeleteVirtualInterfaceRequest request) {request = beforeClientExecution(request);return executeDeleteVirtualInterface(request);}
public Entry getEntry(String name) throws FileNotFoundException {if (excludes.contains(name)) {throw new FileNotFoundException(name);}Entry entry = directory.getEntry(name);return wrapEntry(entry);}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append("[BACKUP]\n");buffer.append("    .backup          = ").append(Integer.toHexString(getBackup())).append("\n");buffer.append("[/BACKUP]\n");return buffer.toString();}
public DeleteVoiceConnectorOriginationResult deleteVoiceConnectorOrigination(DeleteVoiceConnectorOriginationRequest request) {request = beforeClientExecution(request);return executeDeleteVoiceConnectorOrigination(request);}
public Appendable append(char c) {write(c);return this;}
public static long generationFromSegmentsFileName(String fileName) {if (fileName.equals(OLD_SEGMENTS_GEN)) {throw new IllegalArgumentException("\"" + OLD_SEGMENTS_GEN + "\" is not a valid segment file name since 4.0");} else if (fileName.equals(IndexFileNames.SEGMENTS)) {return 0;} else if (fileName.startsWith(IndexFileNames.SEGMENTS)) {return Long.parseLong(fileName.substring(1+IndexFileNames.SEGMENTS.length()),Character.MAX_RADIX);} else {throw new IllegalArgumentException("fileName \"" + fileName + "\" is not a segments file");}}
public static TagOpt fromOption(String o) {if (o == null || o.length() == 0)return AUTO_FOLLOW;for (TagOpt tagopt : values()) {if (tagopt.option().equals(o))return tagopt;}throw new IllegalArgumentException(MessageFormat.format(JGitText.get().invalidTagOption, o));}
public StartContentModerationResult startContentModeration(StartContentModerationRequest request) {request = beforeClientExecution(request);return executeStartContentModeration(request);}
public static String quoteReplacement(String s) {StringBuilder result = new StringBuilder(s.length());for (int i = 0; i < s.length(); i++) {char c = s.charAt(i);if (c == '\\' || c == '$') {result.append('\\');}result.append(c);}return result.toString();}
public final void set(V newValue) {value = newValue;}
public QueryParserTokenManager(CharStream stream){input_stream = stream;}
public long valueFor(double elapsed) {double val;if (modBy == 0)val = elapsed / factor;elseval = elapsed / factor % modBy;if (type == '0')return Math.round(val);elsereturn (long) val;}
public LongBuffer get(long[] dst, int dstOffset, int longCount) {byteBuffer.limit(limit * SizeOf.LONG);byteBuffer.position(position * SizeOf.LONG);if (byteBuffer instanceof DirectByteBuffer) {((DirectByteBuffer) byteBuffer).get(dst, dstOffset, longCount);} else {((HeapByteBuffer) byteBuffer).get(dst, dstOffset, longCount);}this.position += longCount;return this;}
public void removeErrorListeners() {_listeners.clear();}
public CommonTokenStream(TokenSource tokenSource, int channel) {this(tokenSource);this.channel = channel;}
public ListObjectPoliciesResult listObjectPolicies(ListObjectPoliciesRequest request) {request = beforeClientExecution(request);return executeListObjectPolicies(request);}
public ObjectToPack(AnyObjectId src, int type) {super(src);flags = type << TYPE_SHIFT;}
public int stem(char s[], int len) {int numVowels = numVowels(s, len);for (int i = 0; i < affixes.length; i++) {Affix affix = affixes[i];if (numVowels > affix.vc && len >= affix.affix.length + 3 && endsWith(s, len, affix.affix)) {len -= affix.affix.length;return affix.palatalizes ? unpalatalize(s, len) : len;}}return len;}
public void recover(Parser recognizer, RecognitionException e) {if ( lastErrorIndex==recognizer.getInputStream().index() &&lastErrorStates != null &&lastErrorStates.contains(recognizer.getState()) ) {recognizer.consume();}lastErrorIndex = recognizer.getInputStream().index();if ( lastErrorStates==null ) lastErrorStates = new IntervalSet();lastErrorStates.add(recognizer.getState());IntervalSet followSet = getErrorRecoverySet(recognizer);consumeUntil(recognizer, followSet);}
public String toFormulaString() {String value = field_3_string;int len = value.length();StringBuilder sb = new StringBuilder(len + 4);sb.append(FORMULA_DELIMITER);for (int i = 0; i < len; i++) {char c = value.charAt(i);if (c == FORMULA_DELIMITER) {sb.append(FORMULA_DELIMITER);}sb.append(c);}sb.append(FORMULA_DELIMITER);return sb.toString();}
public UnlinkFaceRequest() {super("LinkFace", "2018-07-20", "UnlinkFace");setProtocol(ProtocolType.HTTPS);setMethod(MethodType.POST);}
public ConfigurationOptionSetting(String namespace, String optionName, String value) {setNamespace(namespace);setOptionName(optionName);setValue(value);}
public CharSequence getFully(CharSequence key) {StringBuilder result = new StringBuilder(tries.size() * 2);for (int i = 0; i < tries.size(); i++) {CharSequence r = tries.get(i).getFully(key);if (r == null || (r.length() == 1 && r.charAt(0) == EOM)) {return result;}result.append(r);}return result;}
public DescribeMountTargetSecurityGroupsResult describeMountTargetSecurityGroups(DescribeMountTargetSecurityGroupsRequest request) {request = beforeClientExecution(request);return executeDescribeMountTargetSecurityGroups(request);}
public GetApiMappingResult getApiMapping(GetApiMappingRequest request) {request = beforeClientExecution(request);return executeGetApiMapping(request);}
public HttpRequest(String strUrl) {super(strUrl);}
public MemFuncPtg(int subExprLen) {field_1_len_ref_subexpression = subExprLen;}
public static TermStats[] getHighFreqTerms(IndexReader reader, int numTerms, String field, Comparator<TermStats> comparator) throws Exception {TermStatsQueue tiq = null;if (field != null) {Terms terms = MultiTerms.getTerms(reader, field);if (terms == null) {throw new RuntimeException("field " + field + " not found");}TermsEnum termsEnum = terms.iterator();tiq = new TermStatsQueue(numTerms, comparator);tiq.fill(field, termsEnum);} else {Collection<String> fields = FieldInfos.getIndexedFields(reader);if (fields.size() == 0) {throw new RuntimeException("no fields found for this index");}tiq = new TermStatsQueue(numTerms, comparator);for (String fieldName : fields) {Terms terms = MultiTerms.getTerms(reader, fieldName);if (terms != null) {tiq.fill(fieldName, terms.iterator());}}}TermStats[] result = new TermStats[tiq.size()];int count = tiq.size() - 1;while (tiq.size() != 0) {result[count] = tiq.pop();count--;}return result;}
public DeleteApnsVoipChannelResult deleteApnsVoipChannel(DeleteApnsVoipChannelRequest request) {request = beforeClientExecution(request);return executeDeleteApnsVoipChannel(request);}
public ListFacesResult listFaces(ListFacesRequest request) {request = beforeClientExecution(request);return executeListFaces(request);}
public ShapeFieldCacheDistanceValueSource(SpatialContext ctx,ShapeFieldCacheProvider<Point> provider, Point from, double multiplier) {this.ctx = ctx;this.from = from;this.provider = provider;this.multiplier = multiplier;}
public char get(int index) {checkIndex(index);return sequence.charAt(index);}
public UpdateConfigurationProfileResult updateConfigurationProfile(UpdateConfigurationProfileRequest request) {request = beforeClientExecution(request);return executeUpdateConfigurationProfile(request);}
public DescribeLifecycleHooksResult describeLifecycleHooks(DescribeLifecycleHooksRequest request) {request = beforeClientExecution(request);return executeDescribeLifecycleHooks(request);}
public DescribeHostReservationsResult describeHostReservations(DescribeHostReservationsRequest request) {request = beforeClientExecution(request);return executeDescribeHostReservations(request);}
public static PredictionContext fromRuleContext(ATN atn, RuleContext outerContext) {if ( outerContext==null ) outerContext = RuleContext.EMPTY;if ( outerContext.parent==null || outerContext==RuleContext.EMPTY ) {return PredictionContext.EMPTY;}PredictionContext parent = EMPTY;parent = PredictionContext.fromRuleContext(atn, outerContext.parent);ATNState state = atn.states.get(outerContext.invokingState);RuleTransition transition = (RuleTransition)state.transition(0);return SingletonPredictionContext.create(parent, transition.followState.stateNumber);}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[SXVDEX]\n");buffer.append("    .grbit1 =").append(HexDump.intToHex(_grbit1)).append("\n");buffer.append("    .grbit2 =").append(HexDump.byteToHex(_grbit2)).append("\n");buffer.append("    .citmShow =").append(HexDump.byteToHex(_citmShow)).append("\n");buffer.append("    .isxdiSort =").append(HexDump.shortToHex(_isxdiSort)).append("\n");buffer.append("    .isxdiShow =").append(HexDump.shortToHex(_isxdiShow)).append("\n");buffer.append("    .subtotalName =").append(_subtotalName).append("\n");buffer.append("[/SXVDEX]\n");return buffer.toString();}
public String toString() {StringBuilder r = new StringBuilder();r.append("BlameResult: "); r.append(getResultPath());return r.toString();}
public ListChangeSetsResult listChangeSets(ListChangeSetsRequest request) {request = beforeClientExecution(request);return executeListChangeSets(request);}
public boolean isAllowNonFastForwards() {return allowNonFastForwards;}
public FeatRecord() {futureHeader = new FtrHeader();futureHeader.setRecordType(sid);}
public ShortBuffer put(short c) {throw new ReadOnlyBufferException();}
public void setQuery(CharSequence query) {this.query = query;this.message = new MessageImpl(QueryParserMessages.INVALID_SYNTAX_CANNOT_PARSE, query, "");}
public StashApplyCommand stashApply() {return new StashApplyCommand(repo);}
public Set<String> nameSet() {return Collections.unmodifiableSet(dictionary.values());}
public static int getEffectivePort(String scheme, int specifiedPort) {if (specifiedPort != -1) {return specifiedPort;}if ("http".equalsIgnoreCase(scheme)) {return 80;} else if ("https".equalsIgnoreCase(scheme)) {return 443;} else {return -1;}}
public ListAssessmentTemplatesResult listAssessmentTemplates(ListAssessmentTemplatesRequest request) {request = beforeClientExecution(request);return executeListAssessmentTemplates(request);}
public Cluster restoreFromClusterSnapshot(RestoreFromClusterSnapshotRequest request) {request = beforeClientExecution(request);return executeRestoreFromClusterSnapshot(request);}
public void addShape(HSSFShape shape) {shape.setPatriarch(this.getPatriarch());shape.setParent(this);shapes.add(shape);}
public boolean equals(Object o) {if (this == o) return true;if (o == null || getClass() != o.getClass()) return false;FacetEntry that = (FacetEntry) o;if (count != that.count) return false;if (!value.equals(that.value)) return false;return true;}
public static final int prev(byte[] b, int ptr, char chrA) {if (ptr == b.length)--ptr;while (ptr >= 0) {if (b[ptr--] == chrA)return ptr;}return ptr;}
public final boolean isDeltaRepresentation() {return deltaBase != null;}
public Token emitEOF() {int cpos = getCharPositionInLine();int line = getLine();Token eof = _factory.create(_tokenFactorySourcePair, Token.EOF, null, Token.DEFAULT_CHANNEL, _input.index(), _input.index()-1,line, cpos);emit(eof);return eof;}
public UpdateUserRequest(String userName) {setUserName(userName);}
public RevFilter negate() {return NotRevFilter.create(this);}
public void setTagger(PersonIdent taggerIdent) {tagger = taggerIdent;}
public static BufferSize automatic() {Runtime rt = Runtime.getRuntime();final long max = rt.maxMemory(); final long total = rt.totalMemory(); final long free = rt.freeMemory(); final long totalAvailableBytes = max - total + free;long sortBufferByteSize = free/2;final long minBufferSizeBytes = MIN_BUFFER_SIZE_MB*MB;if (sortBufferByteSize <  minBufferSizeBytes|| totalAvailableBytes > 10 * minBufferSizeBytes) { if (totalAvailableBytes/2 > minBufferSizeBytes) { sortBufferByteSize = totalAvailableBytes/2; } else {sortBufferByteSize = Math.max(ABSOLUTE_MIN_SORT_BUFFER_SIZE, sortBufferByteSize);}}return new BufferSize(Math.min((long)Integer.MAX_VALUE, sortBufferByteSize));}
public static int trimTrailingWhitespace(byte[] raw, int start, int end) {int ptr = end - 1;while (start <= ptr && isWhitespace(raw[ptr]))ptr--;return ptr + 1;}
public TopMarginRecord( RecordInputStream in ) {field_1_margin = in.readDouble();}
public RetrieveEnvironmentInfoRequest(EnvironmentInfoType infoType) {setInfoType(infoType.toString());}
public CreatePlayerSessionsResult createPlayerSessions(CreatePlayerSessionsRequest request) {request = beforeClientExecution(request);return executeCreatePlayerSessions(request);}
public CreateProxySessionResult createProxySession(CreateProxySessionRequest request) {request = beforeClientExecution(request);return executeCreateProxySession(request);}
public int getObjectType() {return type;}
public String getScheme() {return scheme;}
public void characters(char[] ch, int start, int length) {contents.append(ch, start, length);}
public FetchAlbumTagPhotosRequest() {super("CloudPhoto", "2017-07-11", "FetchAlbumTagPhotos", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public DeleteMembersResult deleteMembers(DeleteMembersRequest request) {request = beforeClientExecution(request);return executeDeleteMembers(request);}
public GetContactReachabilityStatusResult getContactReachabilityStatus(GetContactReachabilityStatusRequest request) {request = beforeClientExecution(request);return executeGetContactReachabilityStatus(request);}
@Override public boolean remove(Object o) {return Impl.this.remove(o) != null;}
public E last() {return backingMap.lastKey();}
public CreateStreamingDistributionResult createStreamingDistribution(CreateStreamingDistributionRequest request) {request = beforeClientExecution(request);return executeCreateStreamingDistribution(request);}
public boolean isAbsolute() {return absolute;}
public DisableAddOnResult disableAddOn(DisableAddOnRequest request) {request = beforeClientExecution(request);return executeDisableAddOn(request);}
public DescribeAliasResult describeAlias(DescribeAliasRequest request) {request = beforeClientExecution(request);return executeDescribeAlias(request);}
public void next(int delta) {while (--delta >= 0) {if (currentSubtree != null)ptr += currentSubtree.getEntrySpan();elseptr++;if (eof())break;parseEntry();}}
public RevFilter clone() {return new Binary(a.clone(), b.clone());}
public Reader create(Reader input) {return new PersianCharFilter(input);}
public String option() {return option;}
public String toString() {final StringBuilder sb = new StringBuilder("[");for (Object item : this) {if (sb.length()>1) sb.append(", ");if (item instanceof char[]) {sb.append((char[]) item);} else {sb.append(item);}}return sb.append(']').toString();}
public DescribeSignalingChannelResult describeSignalingChannel(DescribeSignalingChannelRequest request) {request = beforeClientExecution(request);return executeDescribeSignalingChannel(request);}
public AttachStaticIpResult attachStaticIp(AttachStaticIpRequest request) {request = beforeClientExecution(request);return executeAttachStaticIp(request);}
public String toString() {StringBuilder sb = new StringBuilder(64);CellReference crA = new CellReference(_firstRowIndex, _firstColumnIndex, false, false);CellReference crB = new CellReference(_lastRowIndex, _lastColumnIndex, false, false);sb.append(getClass().getName());sb.append(" [").append(crA.formatAsString()).append(':').append(crB.formatAsString()).append("]");return sb.toString();}
public BloomFilteringPostingsFormat(PostingsFormat delegatePostingsFormat,BloomFilterFactory bloomFilterFactory) {super(BLOOM_CODEC_NAME);this.delegatePostingsFormat = delegatePostingsFormat;this.bloomFilterFactory = bloomFilterFactory;}
public ListTemplatesResult listTemplates(ListTemplatesRequest request) {request = beforeClientExecution(request);return executeListTemplates(request);}
public TimerThread(long resolution, Counter counter) {super(THREAD_NAME);this.resolution = resolution;this.counter = counter;this.setDaemon(true);}
public DrawingRecord() {recordData = EMPTY_BYTE_ARRAY;}
public ListDirectoriesResult listDirectories(ListDirectoriesRequest request) {request = beforeClientExecution(request);return executeListDirectories(request);}
public void decode(byte[] blocks, int blocksOffset, int[] values, int valuesOffset, int iterations) {for (int j = 0; j < iterations; ++j) {final byte block = blocks[blocksOffset++];values[valuesOffset++] = (block >>> 7) & 1;values[valuesOffset++] = (block >>> 6) & 1;values[valuesOffset++] = (block >>> 5) & 1;values[valuesOffset++] = (block >>> 4) & 1;values[valuesOffset++] = (block >>> 3) & 1;values[valuesOffset++] = (block >>> 2) & 1;values[valuesOffset++] = (block >>> 1) & 1;values[valuesOffset++] = block & 1;}}
public GroupingSearch disableCaching() {this.maxCacheRAMMB = null;this.maxDocsToCache = null;return this;}
public static int idealByteArraySize(int need) {for (int i = 4; i < 32; i++)if (need <= (1 << i) - 12)return (1 << i) - 12;return need;}
public UpdateAssessmentTargetResult updateAssessmentTarget(UpdateAssessmentTargetRequest request) {request = beforeClientExecution(request);return executeUpdateAssessmentTarget(request);}
public ModifyVolumeResult modifyVolume(ModifyVolumeRequest request) {request = beforeClientExecution(request);return executeModifyVolume(request);}
public Cell merge(Cell m, Cell e) {if (m.cmd == e.cmd && m.ref == e.ref && m.skip == e.skip) {Cell c = new Cell(m);c.cnt += e.cnt;return c;} else {return null;}}
public ByteBuffer read(int length, long position) throws IOException {if(position >= size()) {throw new IndexOutOfBoundsException("Position " + position + " past the end of the file");}ByteBuffer dst;if (writable) {dst = channel.map(FileChannel.MapMode.READ_WRITE, position, length);buffersToClean.add(dst);} else {channel.position(position);dst = ByteBuffer.allocate(length);int worked = IOUtils.readFully(channel, dst);if(worked == -1) {throw new IndexOutOfBoundsException("Position " + position + " past the end of the file");}}dst.position(0);return dst;}
public void respondActivityTaskCompleted(RespondActivityTaskCompletedRequest request) {request = beforeClientExecution(request);executeRespondActivityTaskCompleted(request);}
public synchronized final void incrementProgressBy(int diff) {setProgress(mProgress + diff);}
public MetadataDiff compareMetadata(DirCacheEntry entry) {if (entry.isAssumeValid())return MetadataDiff.EQUAL;if (entry.isUpdateNeeded())return MetadataDiff.DIFFER_BY_METADATA;if (isModeDifferent(entry.getRawMode()))return MetadataDiff.DIFFER_BY_METADATA;int type = mode & FileMode.TYPE_MASK;if (type == FileMode.TYPE_TREE || type == FileMode.TYPE_GITLINK)return MetadataDiff.EQUAL;if (!entry.isSmudged() && entry.getLength() != (int) getEntryLength())return MetadataDiff.DIFFER_BY_METADATA;Instant cacheLastModified = entry.getLastModifiedInstant();Instant fileLastModified = getEntryLastModifiedInstant();if (timestampComparator.compare(cacheLastModified, fileLastModified,getOptions().getCheckStat() == CheckStat.MINIMAL) != 0) {return MetadataDiff.DIFFER_BY_TIMESTAMP;}if (entry.isSmudged()) {return MetadataDiff.SMUDGED;}return MetadataDiff.EQUAL;}
public static NumberRecord convertToNumberRecord(RKRecord rk) {NumberRecord num = new NumberRecord();num.setColumn(rk.getColumn());num.setRow(rk.getRow());num.setXFIndex(rk.getXFIndex());num.setValue(rk.getRKNumber());return num;}
public CharBuffer put(char[] src, int srcOffset, int charCount) {byteBuffer.limit(limit * SizeOf.CHAR);byteBuffer.position(position * SizeOf.CHAR);if (byteBuffer instanceof ReadWriteDirectByteBuffer) {((ReadWriteDirectByteBuffer) byteBuffer).put(src, srcOffset, charCount);} else {((ReadWriteHeapByteBuffer) byteBuffer).put(src, srcOffset, charCount);}this.position += charCount;return this;}
public int getCells() {Iterator<Character> i = cells.keySet().iterator();int size = 0;for (; i.hasNext();) {Character c = i.next();Cell e = at(c);if (e.cmd >= 0 || e.ref >= 0) {size++;}}return size;}
public BeiderMorseFilterFactory(Map<String,String> args) {super(args);NameType nameType = NameType.valueOf(get(args, "nameType", NameType.GENERIC.toString()));RuleType ruleType = RuleType.valueOf(get(args, "ruleType", RuleType.APPROX.toString()));boolean concat = getBoolean(args, "concat", true);engine = new PhoneticEngine(nameType, ruleType, concat);Set<String> langs = getSet(args, "languageSet");languageSet = (null == langs || (1 == langs.size() && langs.contains("auto"))) ? null : LanguageSet.from(langs);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public static double varp(double[] v) {double r = Double.NaN;if (v!=null && v.length > 1) {r = devsq(v) /v.length;}return r;}
public PersianNormalizationFilterFactory(Map<String,String> args) {super(args);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public static WeightedTerm[] getTerms(Query query, boolean prohibited, String fieldName) {HashSet<WeightedTerm> terms = new HashSet<>();Predicate<String> fieldSelector = fieldName == null ? f -> true : fieldName::equals;query.visit(new BoostedTermExtractor(1, terms, prohibited, fieldSelector));return terms.toArray(new WeightedTerm[0]);}
public DeleteDocumentationPartResult deleteDocumentationPart(DeleteDocumentationPartRequest request) {request = beforeClientExecution(request);return executeDeleteDocumentationPart(request);}
public String toString() {StringBuilder sb = new StringBuilder();sb.append("[CHART]\n");sb.append("    .x     = ").append(getX()).append('\n');sb.append("    .y     = ").append(getY()).append('\n');sb.append("    .width = ").append(getWidth()).append('\n');sb.append("    .height= ").append(getHeight()).append('\n');sb.append("[/CHART]\n");return sb.toString();}
public final short get(int index) {checkIndex(index);return backingArray[offset + index];}
public String toString(){return image;}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0, ValueEval arg1) {try {AreaEval reA = evaluateRef(arg0);AreaEval reB = evaluateRef(arg1);AreaEval result = resolveRange(reA, reB);if (result == null) {return ErrorEval.NULL_INTERSECTION;}return result;} catch (EvaluationException e) {return e.getErrorEval();}}
public void clear() {weightBySpanQuery.clear();}
public int findEndOffset(StringBuilder buffer, int start) {if( start > buffer.length() || start < 0 ) return start;bi.setText(buffer.substring(start));return bi.next() + start;}
final public SrndQuery PrimaryQuery() throws ParseException {SrndQuery q;switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {case LPAREN:jj_consume_token(LPAREN);q = FieldsQuery();jj_consume_token(RPAREN);break;case OR:case AND:case W:case N:q = PrefixOperatorQuery();break;case TRUNCQUOTED:case QUOTED:case SUFFIXTERM:case TRUNCTERM:case TERM:q = SimpleTerm();break;default:jj_la1[5] = jj_gen;jj_consume_token(-1);throw new ParseException();}OptionalWeights(q);{if (true) return q;}throw new Error("Missing return statement in function");}
public DeleteApiKeyResult deleteApiKey(DeleteApiKeyRequest request) {request = beforeClientExecution(request);return executeDeleteApiKey(request);}
public InsertTagsRequest() {super("Ots", "2016-06-20", "InsertTags", "ots");setMethod(MethodType.POST);}
public DeleteUserByPrincipalIdResult deleteUserByPrincipalId(DeleteUserByPrincipalIdRequest request) {request = beforeClientExecution(request);return executeDeleteUserByPrincipalId(request);}
public DescribeNetworkInterfacesResult describeNetworkInterfaces(DescribeNetworkInterfacesRequest request) {request = beforeClientExecution(request);return executeDescribeNetworkInterfaces(request);}
public int serialize( int offset, byte[] data, EscherSerializationListener listener ){listener.beforeRecordSerialize( offset, getRecordId(), this );LittleEndian.putShort( data, offset, getOptions() );LittleEndian.putShort( data, offset + 2, getRecordId() );LittleEndian.putInt( data, offset + 4, 8 );LittleEndian.putInt( data, offset + 8, field_1_numShapes );LittleEndian.putInt( data, offset + 12, field_2_lastMSOSPID );listener.afterRecordSerialize( offset + 16, getRecordId(), getRecordSize(), this );return getRecordSize();}
public CreateSecurityConfigurationResult createSecurityConfiguration(CreateSecurityConfigurationRequest request) {request = beforeClientExecution(request);return executeCreateSecurityConfiguration(request);}
public DescribeClientVpnConnectionsResult describeClientVpnConnections(DescribeClientVpnConnectionsRequest request) {request = beforeClientExecution(request);return executeDescribeClientVpnConnections(request);}
public static void fill(double[] array, double value) {for (int i = 0; i < array.length; i++) {array[i] = value;}}
public boolean hasNext() {return nextId < cells.length;}
public PostingsEnum reset(int[] postings) {this.postings = postings;upto = -2;freq = 0;return this;}
public final boolean hasAll(RevFlagSet set) {return (flags & set.mask) == set.mask;}
public ModifyAccountResult modifyAccount(ModifyAccountRequest request) {request = beforeClientExecution(request);return executeModifyAccount(request);}
public Token LT(int k) {lazyInit();if ( k==0 ) return null;if ( k < 0 ) return LB(-k);int i = p + k - 1;sync(i);if ( i >= tokens.size() ) { return tokens.get(tokens.size()-1);}return tokens.get(i);}
public void removeSheet(int sheetIndex) {if (boundsheets.size() > sheetIndex) {records.remove(records.getBspos() - (boundsheets.size() - 1) + sheetIndex);boundsheets.remove(sheetIndex);fixTabIdRecord();}int sheetNum1Based = sheetIndex + 1;for(int i=0; i<getNumNames(); i++) {NameRecord nr = getNameRecord(i);if(nr.getSheetNumber() == sheetNum1Based) {nr.setSheetNumber(0);} else if(nr.getSheetNumber() > sheetNum1Based) {nr.setSheetNumber(nr.getSheetNumber()-1);}}if (linkTable != null) {linkTable.removeSheet(sheetIndex);}}
public void removeName(String name) {int index = getNameIndex(name);removeName(index);}
public boolean equals(final Object o) {if (!(o instanceof Property)) {return false;}final Property p = (Property) o;final Object pValue = p.getValue();final long pId = p.getID();if (id != pId || (id != 0 && !typesAreEqual(type, p.getType()))) {return false;}if (value == null && pValue == null) {return true;}if (value == null || pValue == null) {return false;}final Class<?> valueClass = value.getClass();final Class<?> pValueClass = pValue.getClass();if (!(valueClass.isAssignableFrom(pValueClass)) &&!(pValueClass.isAssignableFrom(valueClass))) {return false;}if (value instanceof byte[]) {byte[] thisVal = (byte[]) value, otherVal = (byte[]) pValue;int len = unpaddedLength(thisVal);if (len != unpaddedLength(otherVal)) {return false;}for (int i=0; i<len; i++) {if (thisVal[i] != otherVal[i]) {return false;}}return true;}return value.equals(pValue);}
public GetRepoBuildListRequest() {super("cr", "2016-06-07", "GetRepoBuildList", "cr");setUriPattern("/repos/[RepoNamespace]/[RepoName]/build");setMethod(MethodType.GET);}
public MessageWriter() {buf = new ByteArrayOutputStream();enc = new OutputStreamWriter(getRawStream(), UTF_8);}
public void append(RecordBase r){_recs.add(r);}
public void close() throws IOException {if (read(skipBuffer) != -1 || actualSize != expectedSize) {throw new CorruptObjectException(MessageFormat.format(JGitText.get().packfileCorruptionDetected,JGitText.get().wrongDecompressedLength));}int used = bAvail - inf.getRemaining();if (0 < used) {onObjectData(src, buf, p, used);use(used);}inf.reset();}
public DescribeModelPackageResult describeModelPackage(DescribeModelPackageRequest request) {request = beforeClientExecution(request);return executeDescribeModelPackage(request);}
public void construct(CellValueRecordInterface rec, RecordStream rs, SharedValueManager sfh) {if (rec instanceof FormulaRecord) {FormulaRecord formulaRec = (FormulaRecord)rec;StringRecord cachedText;Class<? extends Record> nextClass = rs.peekNextClass();if (nextClass == StringRecord.class) {cachedText = (StringRecord) rs.getNext();} else {cachedText = null;}insertCell(new FormulaRecordAggregate(formulaRec, cachedText, sfh));} else {insertCell(rec);}}
public Decompressor clone() {return new DeflateDecompressor();}
public UpdateS3ResourcesResult updateS3Resources(UpdateS3ResourcesRequest request) {request = beforeClientExecution(request);return executeUpdateS3Resources(request);}
public GroupQueryNode(QueryNode query) {if (query == null) {throw new QueryNodeError(new MessageImpl(QueryParserMessages.PARAMETER_VALUE_NOT_SUPPORTED, "query", "null"));}allocate();setLeaf(false);add(query);}
public CharSequence toQueryString(EscapeQuerySyntax escaper) {StringBuilder path = new StringBuilder();path.append("/").append(getFirstPathElement());for (QueryText pathelement : getPathElements(1)) {CharSequence value = escaper.escape(pathelement.value, Locale.getDefault(), Type.STRING);path.append("/\"").append(value).append("\"");}return path.toString();}
public void removeCellComment() {HSSFComment comment = _sheet.findCellComment(_record.getRow(), _record.getColumn());_comment = null;if (null == comment){return;}_sheet.getDrawingPatriarch().removeShape(comment);}
public void reset() {arriving = -1;leaving = -1;}
public ActivateUserResult activateUser(ActivateUserRequest request) {request = beforeClientExecution(request);return executeActivateUser(request);}
public boolean isCharsetDetected() {throw new UnsupportedOperationException();}
public Cluster modifySnapshotCopyRetentionPeriod(ModifySnapshotCopyRetentionPeriodRequest request) {request = beforeClientExecution(request);return executeModifySnapshotCopyRetentionPeriod(request);}
public DeleteClusterSubnetGroupResult deleteClusterSubnetGroup(DeleteClusterSubnetGroupRequest request) {request = beforeClientExecution(request);return executeDeleteClusterSubnetGroup(request);}
public static String decode(byte[] buffer) {return decode(buffer, 0, buffer.length);}
public int getDefaultPort() {return -1;}
public StopTaskResult stopTask(StopTaskRequest request) {request = beforeClientExecution(request);return executeStopTask(request);}
public void seekExact(BytesRef target, TermState otherState) {assert otherState != null && otherState instanceof BlockTermState;assert !doOrd || ((BlockTermState) otherState).ord < numTerms;state.copyFrom(otherState);seekPending = true;indexIsCurrent = false;term.copyBytes(target);}
public SeriesToChartGroupRecord(RecordInputStream in) {field_1_chartGroupIndex = in.readShort();}
public static void writeUnicodeStringFlagAndData(LittleEndianOutput out, String value) {boolean is16Bit = hasMultibyte(value);out.writeByte(is16Bit ? 0x01 : 0x00);if (is16Bit) {putUnicodeLE(value, out);} else {putCompressedUnicode(value, out);}}
public AuthorizeSecurityGroupIngressResult authorizeSecurityGroupIngress(AuthorizeSecurityGroupIngressRequest request) {request = beforeClientExecution(request);return executeAuthorizeSecurityGroupIngress(request);}
public void addFile(String file) {checkFileNames(Collections.singleton(file));setFiles.add(namedForThisSegment(file));}
public void setSize(int width, int height) {mWidth = width;mHeight = height;}
public final void setPrecedenceFilterSuppressed(boolean value) {if (value) {this.reachesIntoOuterContext |= 0x40000000;}else {this.reachesIntoOuterContext &= ~SUPPRESS_PRECEDENCE_FILTER;}}
public IntervalSet LOOK(ATNState s, RuleContext ctx) {return LOOK(s, null, ctx);}
public void serialize(LittleEndianOutput out) {out.writeShort(getOptionFlags());out.writeShort(getRowHeight());}
public Builder(boolean dedup) {this.dedup = dedup;}
public Hashtable(int capacity, float loadFactor) {this(capacity);if (loadFactor <= 0 || Float.isNaN(loadFactor)) {throw new IllegalArgumentException("Load factor: " + loadFactor);}}
public Object get(CharSequence key) {final int bucket = normalCompletion.getBucket(key);return bucket == -1 ? null : Long.valueOf(bucket);}
public ListHyperParameterTuningJobsResult listHyperParameterTuningJobs(ListHyperParameterTuningJobsRequest request) {request = beforeClientExecution(request);return executeListHyperParameterTuningJobs(request);}
public DeleteTableResult deleteTable(String tableName) {return deleteTable(new DeleteTableRequest().withTableName(tableName));}
public final boolean lessThan(TextFragment fragA, TextFragment fragB){if (fragA.getScore() == fragB.getScore())return fragA.fragNum > fragB.fragNum;elsereturn fragA.getScore() < fragB.getScore();}
public void freeBefore(int pos) {assert pos >= 0;assert pos <= nextPos;final int newCount = nextPos - pos;assert newCount <= count: "newCount=" + newCount + " count=" + count;assert newCount <= buffer.length: "newCount=" + newCount + " buf.length=" + buffer.length;count = newCount;}
public UpdateHITTypeOfHITResult updateHITTypeOfHIT(UpdateHITTypeOfHITRequest request) {request = beforeClientExecution(request);return executeUpdateHITTypeOfHIT(request);}
public UpdateRecommenderConfigurationResult updateRecommenderConfiguration(UpdateRecommenderConfigurationRequest request) {request = beforeClientExecution(request);return executeUpdateRecommenderConfiguration(request);}
public int compareTo(BytesRef other) {return Arrays.compareUnsigned(this.bytes, this.offset, this.offset + this.length,other.bytes, other.offset, other.offset + other.length);}
public int stem(char s[], int len) {if (len > 4 && s[len-1] == 's')len--;if (len > 5 &&(endsWith(s, len, "ene") ||  (endsWith(s, len, "ane") &&useNynorsk                 )))return len - 3;if (len > 4 &&(endsWith(s, len, "er") ||   endsWith(s, len, "en") ||   endsWith(s, len, "et") ||   (endsWith(s, len, "ar") &&useNynorsk                 )))return len - 2;if (len > 3)switch(s[len-1]) {case 'a':     case 'e':     return len - 1;}return len;}
public DescribeDBSnapshotsResult describeDBSnapshots(DescribeDBSnapshotsRequest request) {request = beforeClientExecution(request);return executeDescribeDBSnapshots(request);}
public SortedSetDocValuesFacetField(String dim, String label) {super("dummy", TYPE);FacetField.verifyLabel(label);FacetField.verifyLabel(dim);this.dim = dim;this.label = label;}
public CreateDocumentationPartResult createDocumentationPart(CreateDocumentationPartRequest request) {request = beforeClientExecution(request);return executeCreateDocumentationPart(request);}
public String getValue() {return value;}
public ShortBuffer asReadOnlyBuffer() {return duplicate();}
public UpdateDataSourcePermissionsResult updateDataSourcePermissions(UpdateDataSourcePermissionsRequest request) {request = beforeClientExecution(request);return executeUpdateDataSourcePermissions(request);}
public static org.apache.poi.hssf.record.Record createSingleRecord(RecordInputStream in) {I_RecordCreator constructor = _recordCreatorsById.get(Integer.valueOf(in.getSid()));if (constructor == null) {return new UnknownRecord(in);}return constructor.create(in);}
public int getCount() {return mTabs.size();}
public DeleteApplicationReferenceDataSourceResult deleteApplicationReferenceDataSource(DeleteApplicationReferenceDataSourceRequest request) {request = beforeClientExecution(request);return executeDeleteApplicationReferenceDataSource(request);}
public CreateProjectVersionResult createProjectVersion(CreateProjectVersionRequest request) {request = beforeClientExecution(request);return executeCreateProjectVersion(request);}
public IntBuffer slice() {return new ReadOnlyIntArrayBuffer(remaining(), backingArray, offset + position);}
public final byte get() {if (position == limit) {throw new BufferUnderflowException();}return this.block.peekByte(offset + position++);}
public LongBuffer put(int index, long c) {checkIndex(index);backingArray[offset + index] = c;return this;}
public StoredField(String name, float value) {super(name, TYPE);fieldsData = value;}
public IntervalSet getExpectedTokensWithinCurrentRule() {ATN atn = getInterpreter().atn;ATNState s = atn.states.get(getState());return atn.nextTokens(s);}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[FILESHARING]\n");buffer.append("    .readonly       = ").append(getReadOnly() == 1 ? "true" : "false").append("\n");buffer.append("    .password       = ").append(Integer.toHexString(getPassword())).append("\n");buffer.append("    .username       = ").append(getUsername()).append("\n");buffer.append("[/FILESHARING]\n");return buffer.toString();}
public SubmoduleInitCommand(Repository repo) {super(repo);paths = new ArrayList<>();}
public void include(String name, AnyObjectId id) {boolean validRefName = Repository.isValidRefName(name) || Constants.HEAD.equals(name);if (!validRefName)throw new IllegalArgumentException(MessageFormat.format(JGitText.get().invalidRefName, name));if (include.containsKey(name))throw new IllegalStateException(JGitText.get().duplicateRef + name);include.put(name, id.toObjectId());}
public Cluster enableSnapshotCopy(EnableSnapshotCopyRequest request) {request = beforeClientExecution(request);return executeEnableSnapshotCopy(request);}
public ValueFiller getValueFiller() {return new ValueFiller() {private final MutableValueFloat mval = new MutableValueFloat();@Override
public void serialize(LittleEndianOutput out) {out.writeByte(getPane());out.writeShort(getActiveCellRow());out.writeShort(getActiveCellCol());out.writeShort(getActiveCellRef());int nRefs = field_6_refs.length;out.writeShort(nRefs);for (CellRangeAddress8Bit field_6_ref : field_6_refs) {field_6_ref.serialize(out);}}
public static Counter newCounter() {return newCounter(false);}
public boolean get(String name, boolean dflt) {boolean vals[] = (boolean[]) valByRound.get(name);if (vals != null) {return vals[roundNumber % vals.length];}String sval = props.getProperty(name, "" + dflt);if (sval.indexOf(":") < 0) {return Boolean.valueOf(sval).booleanValue();}int k = sval.indexOf(":");String colName = sval.substring(0, k);sval = sval.substring(k + 1);colForValByRound.put(name, colName);vals = propToBooleanArray(sval);valByRound.put(name, vals);return vals[roundNumber % vals.length];}
public void preSerialize(){if(records.getTabpos() > 0) {TabIdRecord tir = ( TabIdRecord ) records.get(records.getTabpos());if(tir._tabids.length < boundsheets.size()) {fixTabIdRecord();}}}
public LimitTokenCountAnalyzer(Analyzer delegate, int maxTokenCount, boolean consumeAllTokens) {super(delegate.getReuseStrategy());this.delegate = delegate;this.maxTokenCount = maxTokenCount;this.consumeAllTokens = consumeAllTokens;}
public ExternalBookBlock(int numberOfSheets) {_externalBookRecord = SupBookRecord.createInternalReferences((short) numberOfSheets);_externalNameRecords = new ExternalNameRecord[0];_crnBlocks = new CRNBlock[0];}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append("[SCENARIOPROTECT]\n");buffer.append("    .protect         = ").append(getProtect()).append("\n");buffer.append("[/SCENARIOPROTECT]\n");return buffer.toString();}
public PushCommand setThin(boolean thin) {checkCallable();this.thin = thin;return this;}
public int compareTo(SearcherTracker other) {return Double.compare(other.recordTimeSec, recordTimeSec);}
public ReverseStringFilter create(TokenStream in) {return new ReverseStringFilter(in);}
public BlockList() {directory = BlockList.<T> newDirectory(256);directory[0] = BlockList.<T> newBlock();tailBlock = directory[0];}
public QueryScorer(WeightedSpanTerm[] weightedTerms) {this.fieldWeightedSpanTerms = new HashMap<>(weightedTerms.length);for (int i = 0; i < weightedTerms.length; i++) {WeightedSpanTerm existingTerm = fieldWeightedSpanTerms.get(weightedTerms[i].term);if ((existingTerm == null) ||(existingTerm.weight < weightedTerms[i].weight)) {fieldWeightedSpanTerms.put(weightedTerms[i].term, weightedTerms[i]);maxTermWeight = Math.max(maxTermWeight, weightedTerms[i].getWeight());}}skipInitExtractor = true;}
public boolean equals(Object _other) {assert neverEquals(_other);if (_other instanceof MergedGroup) {MergedGroup<?> other = (MergedGroup<?>) _other;if (groupValue == null) {return other == null;} else {return groupValue.equals(other);}} else {return false;}}
public final Charset charset() {return cs;}
public DescribeExperimentResult describeExperiment(DescribeExperimentRequest request) {request = beforeClientExecution(request);return executeDescribeExperiment(request);}
public EscherGraphics(HSSFShapeGroup escherGroup, HSSFWorkbook workbook, Color forecolor, float verticalPointsPerPixel ){this.escherGroup = escherGroup;this.workbook = workbook;this.verticalPointsPerPixel = verticalPointsPerPixel;this.verticalPixelsPerPoint = 1 / verticalPointsPerPixel;this.font = new Font("Arial", 0, 10);this.foreground = forecolor;}
public String pattern() {return patternText;}
public DeleteRouteTableResult deleteRouteTable(DeleteRouteTableRequest request) {request = beforeClientExecution(request);return executeDeleteRouteTable(request);}
public AssociateVPCWithHostedZoneResult associateVPCWithHostedZone(AssociateVPCWithHostedZoneRequest request) {request = beforeClientExecution(request);return executeAssociateVPCWithHostedZone(request);}
public PutIntegrationResult putIntegration(PutIntegrationRequest request) {request = beforeClientExecution(request);return executePutIntegration(request);}
public SimpleEntry(K theKey, V theValue) {key = theKey;value = theValue;}
public void decode(byte[] blocks, int blocksOffset, long[] values, int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final long byte0 = blocks[blocksOffset++] & 0xFF;final long byte1 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = (byte0 << 4) | (byte1 >>> 4);final long byte2 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte1 & 15) << 8) | byte2;}}
public DisassociateConnectionFromLagResult disassociateConnectionFromLag(DisassociateConnectionFromLagRequest request) {request = beforeClientExecution(request);return executeDisassociateConnectionFromLag(request);}
public FileMode getOldMode() {return oldMode;}
@Override public String toString() {return m.toString();}
public StopKeyPhrasesDetectionJobResult stopKeyPhrasesDetectionJob(StopKeyPhrasesDetectionJobRequest request) {request = beforeClientExecution(request);return executeStopKeyPhrasesDetectionJob(request);}
public String toString() {return "[Array Formula or Shared Formula]\n" + "row = " + getRow() + "\n" + "col = " + getColumn() + "\n";}
public ListDominantLanguageDetectionJobsResult listDominantLanguageDetectionJobs(ListDominantLanguageDetectionJobsRequest request) {request = beforeClientExecution(request);return executeListDominantLanguageDetectionJobs(request);}
public String toString() {return "slice start=" + start + " length=" + length + " readerIndex=" + readerIndex;}
public static final int parseHexInt4(final byte digit) {final byte r = digits16[digit];if (r < 0)throw new ArrayIndexOutOfBoundsException();return r;}
public Attribute(String name, String value) {setName(name);setValue(value);}
public DescribeStackSetOperationResult describeStackSetOperation(DescribeStackSetOperationRequest request) {request = beforeClientExecution(request);return executeDescribeStackSetOperation(request);}
public HSSFCell getCell(int cellnum) {return getCell(cellnum, book.getMissingCellPolicy());}
public void write(byte[] b) {writeContinueIfRequired(b.length);_ulrOutput.write(b);}
public ResetImageAttributeRequest(String imageId, ResetImageAttributeName attribute) {setImageId(imageId);setAttribute(attribute.toString());}
public void discardResultContents() {resultContents = null;}
public ObjectId getPeeledObjectId() {return getLeaf().getPeeledObjectId();}
public void undeprecateDomain(UndeprecateDomainRequest request) {request = beforeClientExecution(request);executeUndeprecateDomain(request);}
public void write(LittleEndianOutput out) {out.writeByte(sid + getPtgClass());out.writeByte(field_3_string.length()); out.writeByte(_is16bitUnicode ? 0x01 : 0x00);if (_is16bitUnicode) {StringUtil.putUnicodeLE(field_3_string, out);} else {StringUtil.putCompressedUnicode(field_3_string, out);}}
public DeleteQueueResult deleteQueue(String queueUrl) {return deleteQueue(new DeleteQueueRequest().withQueueUrl(queueUrl));}
public void setCheckEofAfterPackFooter(boolean b) {checkEofAfterPackFooter = b;}
public void swap() {final int sBegin = beginA;final int sEnd = endA;beginA = beginB;endA = endB;beginB = sBegin;endB = sEnd;}
public int getPackedGitWindowSize() {return packedGitWindowSize;}
public PutMetricDataResult putMetricData(PutMetricDataRequest request) {request = beforeClientExecution(request);return executePutMetricData(request);}
public GetCelebrityRecognitionResult getCelebrityRecognition(GetCelebrityRecognitionRequest request) {request = beforeClientExecution(request);return executeGetCelebrityRecognition(request);}
public CreateQueueRequest(String queueName) {setQueueName(queueName);}
public Area3DPxg(int externalWorkbookNumber, SheetIdentifier sheetName, AreaReference arearef) {super(arearef);this.externalWorkbookNumber = externalWorkbookNumber;this.firstSheetName = sheetName.getSheetIdentifier().getName();if (sheetName instanceof SheetRangeIdentifier) {this.lastSheetName = ((SheetRangeIdentifier)sheetName).getLastSheetIdentifier().getName();} else {this.lastSheetName = null;}}
public void setBaseline(long clockTime) {t0 = clockTime;timeout = t0 + ticksAllowed;}
public MoveAddressToVpcResult moveAddressToVpc(MoveAddressToVpcRequest request) {request = beforeClientExecution(request);return executeMoveAddressToVpc(request);}
public String toString() {String coll = collectionModel.getName();if (coll != null) {return String.format(Locale.ROOT, "LM %s - %s", getName(), coll);} else {return String.format(Locale.ROOT, "LM %s", getName());}}
public DescribeLagsResult describeLags(DescribeLagsRequest request) {request = beforeClientExecution(request);return executeDescribeLags(request);}
public AreaEval offset(int relFirstRowIx, int relLastRowIx,int relFirstColIx, int relLastColIx) {if (_refEval == null) {return _areaEval.offset(relFirstRowIx, relLastRowIx, relFirstColIx, relLastColIx);}return _refEval.offset(relFirstRowIx, relLastRowIx, relFirstColIx, relLastColIx);}
public ShortBuffer put(short[] src, int srcOffset, int shortCount) {byteBuffer.limit(limit * SizeOf.SHORT);byteBuffer.position(position * SizeOf.SHORT);if (byteBuffer instanceof ReadWriteDirectByteBuffer) {((ReadWriteDirectByteBuffer) byteBuffer).put(src, srcOffset, shortCount);} else {((ReadWriteHeapByteBuffer) byteBuffer).put(src, srcOffset, shortCount);}this.position += shortCount;return this;}
public void initialize(final String cat) {this._cat=cat;}
public void write(int oneByte) throws IOException {out.write(oneByte);written++;}
public DescribeImportImageTasksResult describeImportImageTasks(DescribeImportImageTasksRequest request) {request = beforeClientExecution(request);return executeDescribeImportImageTasks(request);}
public ColumnInfoRecord(RecordInputStream in) {_firstCol = in.readUShort();_lastCol  = in.readUShort();_colWidth = in.readUShort();_xfIndex  = in.readUShort();_options   = in.readUShort();switch(in.remaining()) {case 2: field_6_reserved  = in.readUShort();break;case 1:field_6_reserved  = in.readByte();break;case 0:field_6_reserved  = 0;break;default:throw new RuntimeException("Unusual record size remaining=(" + in.remaining() + ")");}}
public Status(IndexDiff diff) {super();this.diff = diff;hasUncommittedChanges = !diff.getAdded().isEmpty() || !diff.getChanged().isEmpty() || !diff.getRemoved().isEmpty() || !diff.getMissing().isEmpty() || !diff.getModified().isEmpty() || !diff.getConflicting().isEmpty();clean = !hasUncommittedChanges && diff.getUntracked().isEmpty();}
public CreateExperimentResult createExperiment(CreateExperimentRequest request) {request = beforeClientExecution(request);return executeCreateExperiment(request);}
public UnknownRecord clone() {return copy();}
public FloatBuffer slice() {byteBuffer.limit(limit * SizeOf.FLOAT);byteBuffer.position(position * SizeOf.FLOAT);ByteBuffer bb = byteBuffer.slice().order(byteBuffer.order());FloatBuffer result = new FloatToByteBufferAdapter(bb);byteBuffer.clear();return result;}
public DescribeSnapshotSchedulesResult describeSnapshotSchedules(DescribeSnapshotSchedulesRequest request) {request = beforeClientExecution(request);return executeDescribeSnapshotSchedules(request);}
public ListImagesResult listImages(ListImagesRequest request) {request = beforeClientExecution(request);return executeListImages(request);}
public Diff(int ins, int del, int rep, int noop) {INSERT = ins;DELETE = del;REPLACE = rep;NOOP = noop;}
public String toFormulaString(String[] operands){StringBuilder buffer = new StringBuilder();buffer.append(operands[ 0 ]);buffer.append(",");buffer.append(operands[ 1 ]);return buffer.toString();}
public static void setupEnvironment(String[] workbookNames, ForkedEvaluator[] evaluators) {WorkbookEvaluator[] wbEvals = new WorkbookEvaluator[evaluators.length];for (int i = 0; i < wbEvals.length; i++) {wbEvals[i] = evaluators[i]._evaluator;}CollaboratingWorkbooksEnvironment.setup(workbookNames, wbEvals);}
public ListPhotoTagsRequest() {super("CloudPhoto", "2017-07-11", "ListPhotoTags", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public RandomSamplingFacetsCollector(int sampleSize, long seed) {super(false);this.sampleSize = sampleSize;this.random = new XORShift64Random(seed);this.sampledDocs = null;}
public AllocateStaticIpResult allocateStaticIp(AllocateStaticIpRequest request) {request = beforeClientExecution(request);return executeAllocateStaticIp(request);}
public FeatRecord(RecordInputStream in) {futureHeader = new FtrHeader(in);isf_sharedFeatureType = in.readShort();reserved1 = in.readByte();reserved2 = in.readInt();int cref = in.readUShort();cbFeatData = in.readInt();reserved3 = in.readShort();cellRefs = new CellRangeAddress[cref];for(int i=0; i<cellRefs.length; i++) {cellRefs[i] = new CellRangeAddress(in);}switch(isf_sharedFeatureType) {case FeatHdrRecord.SHAREDFEATURES_ISFPROTECTION:sharedFeature = new FeatProtection(in);break;case FeatHdrRecord.SHAREDFEATURES_ISFFEC2:sharedFeature = new FeatFormulaErr2(in);break;case FeatHdrRecord.SHAREDFEATURES_ISFFACTOID:sharedFeature = new FeatSmartTag(in);break;default:logger.log( POILogger.ERROR, "Unknown Shared Feature " + isf_sharedFeatureType + " found!");}}
public RevCommit tryFastForward(RevCommit newCommit) throws IOException,GitAPIException {Ref head = getHead();ObjectId headId = head.getObjectId();if (headId == null)throw new RefNotFoundException(MessageFormat.format(JGitText.get().refNotResolved, Constants.HEAD));RevCommit headCommit = walk.lookupCommit(headId);if (walk.isMergedInto(newCommit, headCommit))return newCommit;String headName = getHeadName(head);return tryFastForward(headName, headCommit, newCommit);}
public CreateSnapshotScheduleResult createSnapshotSchedule(CreateSnapshotScheduleRequest request) {request = beforeClientExecution(request);return executeCreateSnapshotSchedule(request);}
public Record getNext() {if(!hasNext()) {throw new RuntimeException("Attempt to read past end of record stream");}_countRead ++;return _list.get(_nextIndex++);}
public String toString() {return RawParseUtils.decode(buf.toByteArray());}
public ListTablesRequest(String exclusiveStartTableName) {setExclusiveStartTableName(exclusiveStartTableName);}
public EnableAlarmActionsResult enableAlarmActions(EnableAlarmActionsRequest request) {request = beforeClientExecution(request);return executeEnableAlarmActions(request);}
public Builder() {this(true);}
public boolean equals(Object obj) {final State other = (State) obj;return is_final == other.is_final&& Arrays.equals(this.labels, other.labels)&& referenceEquals(this.states, other.states);}
public TokenStream create(TokenStream input) {return new EnglishPossessiveFilter(input);}
public void clearFormatting() {_string = cloneStringIfRequired();_string.clearFormatting();addToSSTIfRequired();}
public int get(int index, long[] arr, int off, int len) {assert len > 0 : "len must be > 0 (got " + len + ")";assert index >= 0 && index < valueCount;len = Math.min(len, valueCount - index);Arrays.fill(arr, off, off + len, 0);return len;}
public DeleteRouteResponseResult deleteRouteResponse(DeleteRouteResponseRequest request) {request = beforeClientExecution(request);return executeDeleteRouteResponse(request);}
public String toPrivateString() {return format(true, false);}
public CreatePresignedDomainUrlResult createPresignedDomainUrl(CreatePresignedDomainUrlRequest request) {request = beforeClientExecution(request);return executeCreatePresignedDomainUrl(request);}
public void write(int oneChar) {doWrite(new char[] { (char) oneChar }, 0, 1);}
public SSTRecord getSSTRecord() {return sstRecord;}
public String toString() {return "term=" + term + ",field=" + field + ",value=" + valueToString() + ",docIDUpto=" + docIDUpto;}
public boolean isSaturated(FuzzySet bloomFilter, FieldInfo fieldInfo) {return bloomFilter.getSaturation() > 0.9f;}
public Builder(boolean ignoreCase) {this.ignoreCase = ignoreCase;}
public String toString() {return getClass().getName()+ "(maxBasicQueries: " + maxBasicQueries+ ", queriesMade: " + queriesMade+ ")";}
public DeleteDataSourceResult deleteDataSource(DeleteDataSourceRequest request) {request = beforeClientExecution(request);return executeDeleteDataSource(request);}
public RebootNodeResult rebootNode(RebootNodeRequest request) {request = beforeClientExecution(request);return executeRebootNode(request);}
public void processChildRecords() {convertRawBytesToEscherRecords();}
public CreateOrUpdateTagsResult createOrUpdateTags(CreateOrUpdateTagsRequest request) {request = beforeClientExecution(request);return executeCreateOrUpdateTags(request);}
public FileSnapshot getSnapShot() {return snapShot;}
public InputStream openResource(String resource) throws IOException {final InputStream stream = (clazz != null) ?clazz.getResourceAsStream(resource) :loader.getResourceAsStream(resource);if (stream == null)throw new IOException("Resource not found: " + resource);return stream;}
public String toString() {StringBuilder sb = new StringBuilder(64);sb.append(getClass().getName()).append(" [");sb.append("sid=").append(HexDump.shortToHex(_sid));sb.append(" size=").append(_data.length);sb.append(" : ").append(HexDump.toHex(_data));sb.append("]\n");return sb.toString();}
public int nextIndex() {return index;}
public CharSequence toQueryString(EscapeQuerySyntax escaper) {if (isDefaultField(this.field)) {return "\"" + getTermEscapeQuoted(escaper) + "\"";} else {return this.field + ":" + "\"" + getTermEscapeQuoted(escaper) + "\"";}}
public CalcModeRecord clone() {return copy();}
public boolean isOutput() {return output;}
public CreateNetworkInterfaceResult createNetworkInterface(CreateNetworkInterfaceRequest request) {request = beforeClientExecution(request);return executeCreateNetworkInterface(request);}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_password);}
public StopDominantLanguageDetectionJobResult stopDominantLanguageDetectionJob(StopDominantLanguageDetectionJobRequest request) {request = beforeClientExecution(request);return executeStopDominantLanguageDetectionJob(request);}
public ECSMetadataServiceCredentialsFetcher withConnectionTimeout(int milliseconds) {this.connectionTimeoutInMilliseconds = milliseconds;return this;}
public GetGatewayGroupResult getGatewayGroup(GetGatewayGroupRequest request) {request = beforeClientExecution(request);return executeGetGatewayGroup(request);}
public FloatBuffer slice() {return new ReadOnlyFloatArrayBuffer(remaining(), backingArray, offset + position);}
public static String join(Collection<String> parts, String separator,String lastSeparator) {StringBuilder sb = new StringBuilder();int i = 0;int lastIndex = parts.size() - 1;for (String part : parts) {sb.append(part);if (i == lastIndex - 1) {sb.append(lastSeparator);} else if (i != lastIndex) {sb.append(separator);}i++;}return sb.toString();}
public String toString() {return "(" + a.toString() + " AND " + b.toString() + ")"; }
public ListSubscriptionsByTopicRequest(String topicArn, String nextToken) {setTopicArn(topicArn);setNextToken(nextToken);}
public byte readByte() {return bytes[pos--];}
public TerminateClientVpnConnectionsResult terminateClientVpnConnections(TerminateClientVpnConnectionsRequest request) {request = beforeClientExecution(request);return executeTerminateClientVpnConnections(request);}
public ReceiveMessageRequest(String queueUrl) {setQueueUrl(queueUrl);}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_barSpace);out.writeShort(field_2_categorySpace);out.writeShort(field_3_formatFlags);}
public Object common(Object output1, Object output2) {return outputs.common((T) output1, (T) output2);}
public CreateVariableResult createVariable(CreateVariableRequest request) {request = beforeClientExecution(request);return executeCreateVariable(request);}
public static final int match(byte[] b, int ptr, byte[] src) {if (ptr + src.length > b.length)return -1;for (int i = 0; i < src.length; i++, ptr++)if (b[ptr] != src[i])return -1;return ptr;}
public int fillFields(byte[] data, int offset, EscherRecordFactory recordFactory) {int bytesRemaining = readHeader( data, offset );int pos            = offset + 8;int size           = 0;field_1_rectX1 =  LittleEndian.getInt( data, pos + size );size+=4;field_2_rectY1 =  LittleEndian.getInt( data, pos + size );size+=4;field_3_rectX2 =  LittleEndian.getInt( data, pos + size );size+=4;field_4_rectY2 =  LittleEndian.getInt( data, pos + size );size+=4;bytesRemaining -= size;if (bytesRemaining != 0) {throw new RecordFormatException("Expected no remaining bytes but got " + bytesRemaining);}return 8 + size + bytesRemaining;}
public CreateCloudFrontOriginAccessIdentityResult createCloudFrontOriginAccessIdentity(CreateCloudFrontOriginAccessIdentityRequest request) {request = beforeClientExecution(request);return executeCreateCloudFrontOriginAccessIdentity(request);}
public boolean isNamespaceAware() {return getFeature (XmlPullParser.FEATURE_PROCESS_NAMESPACES);}
public void setOverridable(boolean on) {overridable = on;}
public String getClassName() {return className;}
public synchronized DirectoryReader getIndexReader() {if (indexReader != null) {indexReader.incRef();}return indexReader;}
public int indexOfKey(int key) {return binarySearch(mKeys, 0, mSize, key);}
public BlankRecord(RecordInputStream in) {field_1_row = in.readUShort();field_2_col = in.readShort();field_3_xf  = in.readShort();}
public long length() {return length;}
public PasswordRecord(RecordInputStream in) {field_1_password = in.readShort();}
public HashMap(int capacity, float loadFactor) {this(capacity);if (loadFactor <= 0 || Float.isNaN(loadFactor)) {throw new IllegalArgumentException("Load factor: " + loadFactor);}}
public void run() {long lastReopenStartNS = System.nanoTime();while (!finish) {while (!finish) {reopenLock.lock();try {boolean hasWaiting = waitingGen > searchingGen;final long nextReopenStartNS = lastReopenStartNS + (hasWaiting ? targetMinStaleNS : targetMaxStaleNS);final long sleepNS = nextReopenStartNS - System.nanoTime();if (sleepNS > 0) {reopenCond.awaitNanos(sleepNS);} else {break;}} catch (InterruptedException ie) {Thread.currentThread().interrupt();return;} finally {reopenLock.unlock();}}if (finish) {break;}lastReopenStartNS = System.nanoTime();refreshStartGen = writer.getMaxCompletedSequenceNumber();try {manager.maybeRefreshBlocking();} catch (IOException ioe) {throw new RuntimeException(ioe);}}}
public DeleteLoginProfileRequest(String userName) {setUserName(userName);}
public E pollFirst() {return (size == 0) ? null : removeFirstImpl();}
public CreatePhotoRequest() {super("CloudPhoto", "2017-07-11", "CreatePhoto", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public String getName() {return "resolve"; }
public int findEndOffset(StringBuilder buffer, int start) {if( start > buffer.length() || start < 0 ) return start;int offset, count = maxScan;for( offset = start; offset < buffer.length() && count > 0; count-- ){if( boundaryChars.contains( buffer.charAt( offset ) ) ) return offset;offset++;}return start;}
public void setObjectChecker(ObjectChecker oc) {objCheck = oc;}
public BaseRef(AreaEval ae) {_refEval = null;_areaEval = ae;_firstRowIndex = ae.getFirstRow();_firstColumnIndex = ae.getFirstColumn();_height = ae.getLastRow() - ae.getFirstRow() + 1;_width = ae.getLastColumn() - ae.getFirstColumn() + 1;}
public CreateVpcEndpointResult createVpcEndpoint(CreateVpcEndpointRequest request) {request = beforeClientExecution(request);return executeCreateVpcEndpoint(request);}
public DeregisterWorkspaceDirectoryResult deregisterWorkspaceDirectory(DeregisterWorkspaceDirectoryRequest request) {request = beforeClientExecution(request);return executeDeregisterWorkspaceDirectory(request);}
public ChartFRTInfoRecord(RecordInputStream in) {rt = in.readShort();grbitFrt = in.readShort();verOriginator = in.readByte();verWriter = in.readByte();int cCFRTID = in.readShort();rgCFRTID = new CFRTID[cCFRTID];for (int i = 0; i < cCFRTID; i++) {rgCFRTID[i] = new CFRTID(in);}}
public Merger newMerger(Repository db) {return new OneSide(db, treeIndex);}
public CreateDataSourceFromRedshiftResult createDataSourceFromRedshift(CreateDataSourceFromRedshiftRequest request) {request = beforeClientExecution(request);return executeCreateDataSourceFromRedshift(request);}
public void clearDFA() {for (int d = 0; d < decisionToDFA.length; d++) {decisionToDFA[d] = new DFA(atn.getDecisionState(d), d);}}
public void removeName(String name) {int index = getNameIndex(name);removeName(index);}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append( "[RightMargin]\n" );buffer.append( "    .margin               = " ).append( " (" ).append( getMargin() ).append( " )\n" );buffer.append( "[/RightMargin]\n" );return buffer.toString();}
public RefreshAllRecord clone() {return copy();}
public StandardQueryNodeProcessorPipeline(QueryConfigHandler queryConfig) {super(queryConfig);add(new WildcardQueryNodeProcessor());add(new MultiFieldQueryNodeProcessor());add(new FuzzyQueryNodeProcessor());add(new RegexpQueryNodeProcessor());add(new MatchAllDocsQueryNodeProcessor());add(new OpenRangeQueryNodeProcessor());add(new PointQueryNodeProcessor());add(new PointRangeQueryNodeProcessor());add(new TermRangeQueryNodeProcessor());add(new AllowLeadingWildcardProcessor());add(new AnalyzerQueryNodeProcessor());add(new PhraseSlopQueryNodeProcessor());add(new BooleanQuery2ModifierNodeProcessor());add(new NoChildOptimizationQueryNodeProcessor());add(new RemoveDeletedQueryNodesProcessor());add(new RemoveEmptyNonLeafQueryNodeProcessor());add(new BooleanSingleChildOptimizationQueryNodeProcessor());add(new DefaultPhraseSlopQueryNodeProcessor());add(new BoostQueryNodeProcessor());add(new MultiTermRewriteMethodProcessor());}
public String formatAsString(String sheetName, boolean useAbsoluteAddress) {StringBuilder sb = new StringBuilder();if (sheetName != null) {sb.append(SheetNameFormatter.format(sheetName));sb.append("!");}CellReference cellRefFrom = new CellReference(getFirstRow(), getFirstColumn(),useAbsoluteAddress, useAbsoluteAddress);CellReference cellRefTo = new CellReference(getLastRow(), getLastColumn(),useAbsoluteAddress, useAbsoluteAddress);sb.append(cellRefFrom.formatAsString());if(!cellRefFrom.equals(cellRefTo)|| isFullColumnRange() || isFullRowRange()){sb.append(':');sb.append(cellRefTo.formatAsString());}return sb.toString();}
public ByteBuffer put(int index, byte value) {throw new ReadOnlyBufferException();}
public void mode(int m) {_mode = m;}
public ShortBuffer slice() {return new ReadWriteShortArrayBuffer(remaining(), backingArray, offset + position);}
public void set(int index, long n) {if (count < index)throw new ArrayIndexOutOfBoundsException(index);else if (count == index)add(n);elseentries[index] = n;}
public ByteBuffer putFloat(float value) {throw new ReadOnlyBufferException();}
public static double max(double[] values) {double max = Double.NEGATIVE_INFINITY;for (double value : values) {max = Math.max(max, value);}return max;}
public UpdateRepoWebhookRequest() {super("cr", "2016-06-07", "UpdateRepoWebhook", "cr");setUriPattern("/repos/[RepoNamespace]/[RepoName]/webhooks/[WebhookId]");setMethod(MethodType.POST);}
public DeleteAttributesRequest(String domainName, String itemName, java.util.List<Attribute> attributes, UpdateCondition expected) {setDomainName(domainName);setItemName(itemName);setAttributes(attributes);setExpected(expected);}
public String toString() {StringBuilder sb = new StringBuilder();sb.append("[SXPI]\n");for (int i = 0; i < _fieldInfos.length; i++) {sb.append("    item[").append(i).append("]=");_fieldInfos[i].appendDebugInfo(sb);sb.append('\n');}sb.append("[/SXPI]\n");return sb.toString();}
public boolean isSuccessful() {if (mergeResult != null)return mergeResult.getMergeStatus().isSuccessful();else if (rebaseResult != null)return rebaseResult.getStatus().isSuccessful();return true;}
public void setBytesValue(byte[] value) {setBytesValue(new BytesRef(value));}
public DescribeConnectionsResult describeConnections(DescribeConnectionsRequest request) {request = beforeClientExecution(request);return executeDescribeConnections(request);}
public DeletePhotosRequest() {super("CloudPhoto", "2017-07-11", "DeletePhotos", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public void add(E object) {iterator.add(object);subList.sizeChanged(true);end++;}
public static ByteBuffer allocate(int capacity) {if (capacity < 0) {throw new IllegalArgumentException();}return new ReadWriteHeapByteBuffer(capacity);}
public SrndQuery getSubQuery(int qn) {return queries.get(qn);}
public float currentScore(int docId, String field, int start, int end, int numPayloadsSeen, float currentScore, float currentPayloadScore) {if (numPayloadsSeen == 0) {return currentPayloadScore;} else {return Math.min(currentPayloadScore, currentScore);}}
public String toString(){StringBuilder sb = new StringBuilder();sb.append("[BLANK]\n");sb.append("    row= ").append(HexDump.shortToHex(getRow())).append("\n");sb.append("    col= ").append(HexDump.shortToHex(getColumn())).append("\n");sb.append("    xf = ").append(HexDump.shortToHex(getXFIndex())).append("\n");sb.append("[/BLANK]\n");return sb.toString();}
public DescribeLogPatternResult describeLogPattern(DescribeLogPatternRequest request) {request = beforeClientExecution(request);return executeDescribeLogPattern(request);}
public RegisterTransitGatewayMulticastGroupMembersResult registerTransitGatewayMulticastGroupMembers(RegisterTransitGatewayMulticastGroupMembersRequest request) {request = beforeClientExecution(request);return executeRegisterTransitGatewayMulticastGroupMembers(request);}
public GetPhoneNumberSettingsResult getPhoneNumberSettings(GetPhoneNumberSettingsRequest request) {request = beforeClientExecution(request);return executeGetPhoneNumberSettings(request);}
public ObjectId getData() {return data;}
public boolean isDirect() {return false;}
public DeleteServerCertificateRequest(String serverCertificateName) {setServerCertificateName(serverCertificateName);}
public StringBuffer append(double d) {RealToString.getInstance().appendDouble(this, d);return this;}
public GetEvaluationResult getEvaluation(GetEvaluationRequest request) {request = beforeClientExecution(request);return executeGetEvaluation(request);}
public LinkedDataRecord getDataName(){return dataName;}
public boolean find(int start) {findPos = start;if (findPos < regionStart) {findPos = regionStart;} else if (findPos >= regionEnd) {matchFound = false;return false;}matchFound = findImpl(address, input, findPos, matchOffsets);if (matchFound) {findPos = matchOffsets[1];}return matchFound;}
public GetLifecyclePolicyPreviewResult getLifecyclePolicyPreview(GetLifecyclePolicyPreviewRequest request) {request = beforeClientExecution(request);return executeGetLifecyclePolicyPreview(request);}
public SinglePositionTokenStream(String word) {termAtt = addAttribute(CharTermAttribute.class);posIncrAtt = addAttribute(PositionIncrementAttribute.class);this.word = word;returned = true;}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_print_gridlines);}
public String toString() {final StringBuilder s = new StringBuilder();s.append(Constants.typeString(getType()));s.append(' ');s.append(name());s.append(' ');s.append(commitTime);s.append(' ');appendCoreFlags(s);return s.toString();}
public LsRemoteCommand setRemote(String remote) {checkCallable();this.remote = remote;return this;}
public void collapseRow(int rowNumber) {int startRow = findStartOfRowOutlineGroup(rowNumber);RowRecord rowRecord = getRow(startRow);int nextRowIx = writeHidden(rowRecord, startRow);RowRecord row = getRow(nextRowIx);if (row == null) {row = createRow(nextRowIx);insertRow(row);}row.setColapsed(true);}
public AssociateSkillGroupWithRoomResult associateSkillGroupWithRoom(AssociateSkillGroupWithRoomRequest request) {request = beforeClientExecution(request);return executeAssociateSkillGroupWithRoom(request);}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[SERIESLIST]\n");buffer.append("    .seriesNumbers= ").append(" (").append( Arrays.toString(getSeriesNumbers()) ).append(" )");buffer.append("\n");buffer.append("[/SERIESLIST]\n");return buffer.toString();}
public QueryConfigHandler getQueryConfigHandler() {return this.queryConfig;}
public String getClassArg() {if (null != originalArgs) {String className = originalArgs.get(CLASS_NAME);if (null != className) {return className;}}return getClass().getName();}
