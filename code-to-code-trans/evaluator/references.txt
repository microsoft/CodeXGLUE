public override void Serialize(ILittleEndianOutput out1){out1.WriteShort(field_1_vcenter);}
public virtual void AddAll(NGit.Util.BlockList<T> src){if (src.size == 0){return;}int srcDirIdx = 0;for (; srcDirIdx < src.tailDirIdx; srcDirIdx++){AddAll(src.directory[srcDirIdx], 0, BLOCK_SIZE);}if (src.tailBlkIdx != 0){AddAll(src.tailBlock, 0, src.tailBlkIdx);}}
public override void WriteByte(byte b){if (outerInstance.upto == outerInstance.blockSize){if (outerInstance.currentBlock != null){outerInstance.blocks.Add(outerInstance.currentBlock);outerInstance.blockEnd.Add(outerInstance.upto);}outerInstance.currentBlock = new byte[outerInstance.blockSize];outerInstance.upto = 0;}outerInstance.currentBlock[outerInstance.upto++] = (byte)b;}
public virtual ObjectId GetObjectId(){return objectId;}
public virtual DeleteDomainEntryResponse DeleteDomainEntry(DeleteDomainEntryRequest request){var options = new InvokeOptions();options.RequestMarshaller = DeleteDomainEntryRequestMarshaller.Instance;options.ResponseUnmarshaller = DeleteDomainEntryResponseUnmarshaller.Instance;return Invoke<DeleteDomainEntryResponse>(request, options);}
public virtual long RamBytesUsed(){return fst == null ? 0 : fst.GetSizeInBytes();}
public string GetFullMessage(){byte[] raw = buffer;int msgB = RawParseUtils.TagMessage(raw, 0);if (msgB < 0){return string.Empty;}Encoding enc = RawParseUtils.ParseEncoding(raw);return RawParseUtils.Decode(enc, raw, msgB, raw.Length);}
public POIFSFileSystem(){HeaderBlock headerBlock = new HeaderBlock(bigBlockSize);_property_table = new PropertyTable(headerBlock);_documents      = new ArrayList();_root           = null;}
public void Init(int address){slice = pool.Buffers[address >> ByteBlockPool.BYTE_BLOCK_SHIFT];Debug.Assert(slice != null);upto = address & ByteBlockPool.BYTE_BLOCK_MASK;offset0 = address;Debug.Assert(upto < slice.Length);}
public virtual NGit.Api.SubmoduleAddCommand SetPath(string path){this.path = path;return this;}
