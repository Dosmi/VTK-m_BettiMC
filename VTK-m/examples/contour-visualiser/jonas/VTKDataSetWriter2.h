//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#pragma once


#include <vtkm/cont/DataSet.h>

#include <vtkm/io/vtkm_io_export.h>
#include <vtkm/io/VTKDataSetWriter.h>

#include <string>

struct VTKM_IO_EXPORT VTKDataSetWriter2
{
public:
  VTKDataSetWriter2(const char* fileName);
  VTKDataSetWriter2(const std::string& fileName);

  void WriteDataSet(const vtkm::cont::DataSet& dataSet) const;

  /// \brief Get whether the file will be written in ASCII or binary format.
  ///
  vtkm::io::FileType GetFileType() const;

  /// \{
  /// \brief Set whether the file will be written in ASCII or binary format.
  void SetFileType(vtkm::io::FileType type);
  void SetFileTypeToAscii() { this->SetFileType(vtkm::io::FileType::ASCII); }
  void SetFileTypeToBinary() { this->SetFileType(vtkm::io::FileType::BINARY); }
  /// \}

private:
  std::string FileName;
  vtkm::io::FileType FileType = vtkm::io::FileType::ASCII;

};
